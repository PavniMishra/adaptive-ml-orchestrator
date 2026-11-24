"""Real-data ML workloads for Adaptive ML Workload Orchestrator.

SMS data is downloaded from the UCI SMS Spam Collection on first execution.
Iris is bundled with scikit-learn. Movie genre classification requires a local
CSV at backend/data/movie_genres.csv with columns: plot, genres. The genres
column must contain pipe-separated labels, e.g. "Drama|Romance".
"""
import csv
import io
import os
import threading
import time
import urllib.request
import zipfile
from functools import lru_cache

import numpy as np
import psutil
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

import models

RANDOM_STATE = 42
N_ESTIMATORS = 100
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SMS_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SUBTYPE_TO_DATASET = {
    "binary": "SMS Spam Collection",
    "multiclass": "Iris Flower Dataset",
    "multilabel": "Movie Genre Dataset",
}


@lru_cache(maxsize=1)
def _load_sms_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    local_path = os.path.join(DATA_DIR, "SMSSpamCollection")
    if not os.path.exists(local_path):
        try:
            request = urllib.request.Request(SMS_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                member = next(n for n in archive.namelist() if n.endswith("SMSSpamCollection"))
                with open(local_path, "wb") as out:
                    out.write(archive.read(member))
        except Exception as exc:
            raise RuntimeError(
                "Could not download the UCI SMS Spam Collection. Check internet access "
                "or place the extracted SMSSpamCollection file in backend/data/."
            ) from exc

    texts, labels = [], []
    with open(local_path, "r", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 2 and row[0] in ("ham", "spam"):
                labels.append(row[0])
                texts.append("\t".join(row[1:]).strip())
    if len(texts) < 20:
        raise RuntimeError("The SMS dataset file is empty or has an unexpected format.")
    return np.asarray(texts, dtype=object), np.asarray(labels, dtype=object)


@lru_cache(maxsize=1)
def _load_iris_data():
    data = load_iris()
    return data.data, data.target, list(data.target_names), list(data.feature_names)


@lru_cache(maxsize=1)
def _load_movie_data():
    path = os.path.join(DATA_DIR, "movie_genres.csv")
    if not os.path.exists(path):
        raise RuntimeError(
            "Movie genre dataset is missing. Add backend/data/movie_genres.csv "
            "with columns 'plot' and 'genres'; genres must be pipe-separated."
        )
    plots, genre_rows = [], []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"plot", "genres"}.issubset(set(reader.fieldnames)):
            raise RuntimeError("movie_genres.csv must contain 'plot' and 'genres' columns.")
        for row in reader:
            plot = (row.get("plot") or "").strip()
            genres = [g.strip() for g in (row.get("genres") or "").split("|") if g.strip()]
            if plot and genres:
                plots.append(plot)
                genre_rows.append(genres)
    if len(plots) < 30:
        raise RuntimeError("movie_genres.csv needs at least 30 valid labelled movie rows.")
    mlb = MultiLabelBinarizer()
    targets = mlb.fit_transform(genre_rows)
    if targets.shape[1] < 2:
        raise RuntimeError("Movie genre CSV must contain at least two distinct genres.")
    return np.asarray(plots, dtype=object), targets, list(mlb.classes_)


def dataset_specs():
    """Metadata for the three real-data tasks; does not download files at startup."""
    return [
        {"name": SUBTYPE_TO_DATASET["binary"], "subtype": "binary", "size_mb": 0.5,
         "n_features": 5000, "n_classes": 2, "n_samples": 5574},
        {"name": SUBTYPE_TO_DATASET["multiclass"], "subtype": "multiclass", "size_mb": 0.01,
         "n_features": 4, "n_classes": 3, "n_samples": 150},
        {"name": SUBTYPE_TO_DATASET["multilabel"], "subtype": "multilabel", "size_mb": 10.0,
         "n_features": 5000, "n_classes": 5, "n_samples": 0},
    ]


def ensure_datasets(db):
    """Create/update the three dataset metadata rows without deleting old records."""
    ids = {}
    specs = {item["subtype"]: item for item in dataset_specs()}
    for subtype, name in SUBTYPE_TO_DATASET.items():
        spec = specs[subtype]
        ds = db.query(models.Dataset).filter(models.Dataset.name == name).first()
        if ds is None:
            ds = models.Dataset(name=name, size_mb=spec["size_mb"],
                                n_features=spec["n_features"], n_classes=spec["n_classes"])
            db.add(ds)
            db.flush()
        else:
            ds.size_mb = spec["size_mb"]
            ds.n_features = spec["n_features"]
            ds.n_classes = spec["n_classes"]
        ids[subtype] = ds.dataset_id
    db.commit()
    return ids


class _PeakRamSampler(threading.Thread):
    def __init__(self, process):
        super().__init__(daemon=True)
        self.process = process
        self.peak = process.memory_info().rss
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except psutil.Error:
                pass
            time.sleep(0.01)

    def stop(self):
        self._halt.set()
        self.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def _sample_indices(length, fraction):
    count = max(20, int(length * fraction))
    count = min(length, count)
    return np.random.RandomState(RANDOM_STATE).choice(length, size=count, replace=False)


def run_classification(subtype: str, data_fraction: float, user_input):
    """Train/evaluate and predict one user example inside the scheduled worker."""
    if subtype not in SUBTYPE_TO_DATASET:
        raise ValueError(f"Unknown subtype: {subtype}")
    if not 0 < float(data_fraction) <= 1:
        raise ValueError("data_fraction must be between 0 and 1")

    process = psutil.Process()
    ram_before = process.memory_info().rss
    cpu_before = sum(process.cpu_times()[:2])
    sampler = _PeakRamSampler(process)
    sampler.start()
    t0 = time.perf_counter()

    if subtype == "binary":
        X, y = _load_sms_data()
        idx = _sample_indices(len(X), data_fraction)
        X, y = X[idx], y[idx]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        model = Pipeline([("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
                          ("classifier", MultinomialNB())])
        prediction_input = str(user_input.get("sms", "")).strip()
        if not prediction_input:
            raise ValueError("Enter an SMS message before submitting the task.")
        metric_name = "Accuracy"

    elif subtype == "multiclass":
        X, y, target_names, feature_names = _load_iris_data()
        idx = _sample_indices(len(X), data_fraction)
        X, y = X[idx], y[idx]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=1, random_state=RANDOM_STATE)
        try:
            values = [float(user_input[key]) for key in
                      ("sepal_length", "sepal_width", "petal_length", "petal_width")]
        except (KeyError, TypeError, ValueError):
            raise ValueError("Enter all four Iris measurements as numbers.")
        if any(v <= 0 for v in values):
            raise ValueError("Iris measurements must be greater than zero.")
        prediction_input = np.asarray(values, dtype=float).reshape(1, -1)
        metric_name = "Accuracy"

    else:
        X, y, genre_names = _load_movie_data()
        idx = _sample_indices(len(X), data_fraction)
        X, y = X[idx], y[idx]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        model = Pipeline([("tfidf", TfidfVectorizer(stop_words="english", max_features=5000,
                                                    ngram_range=(1, 2))),
                          ("classifier", OneVsRestClassifier(LogisticRegression(max_iter=1000)))])
        prediction_input = str(user_input.get("description", "")).strip()
        if len(prediction_input) < 10:
            raise ValueError("Enter a movie description containing at least 10 characters.")
        metric_name = "Micro F1-score"

    t_fit = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t_fit

    
    t_pred = time.perf_counter()

    # Use threshold 0.30 only for multilabel classification
    if subtype == "multilabel":
        threshold = 0.30
        probabilities_test = model.predict_proba(X_test)
        y_pred = (probabilities_test >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    if subtype == "binary":
        score = accuracy_score(y_test, y_pred)
        prediction = str(model.predict([prediction_input])[0]).title()

    elif subtype == "multiclass":
        score = accuracy_score(y_test, y_pred)
        predicted_index = int(model.predict(prediction_input)[0])
        prediction = target_names[predicted_index].title()

    else:
        score = f1_score(
            y_test, y_pred,
            average="micro",
            zero_division=0
        )

        # Predict genres for the user's movie description
        probabilities = model.predict_proba([prediction_input])[0]

        print("\nGENRE PREDICTION PROBABILITIES:")

        for genre, probability in zip(genre_names, probabilities):
            print(f"{genre}: {probability:.4f}")

        predicted = (probabilities >= threshold).astype(int)

        prediction = [
            genre_names[i]
            for i, value in enumerate(predicted)
            if value == 1
        ]

        if not prediction:
            prediction = ["No genre crossed the prediction threshold"]

    predict_time = time.perf_counter() - t_pred

    duration = time.perf_counter() - t0
    cpu_used = sum(process.cpu_times()[:2]) - cpu_before
    sampler.stop()
    cpu_count = psutil.cpu_count() or 1
    cpu_share = cpu_used / duration / cpu_count * 100 if duration > 0 else 0.0

    return {
        "cpu": round(cpu_share, 2),
        "ram_mb": round(max(0, sampler.peak - ram_before) / 1e6, 2),
        "duration": round(duration, 4),
        "fit_time": round(fit_time, 4),
        "predict_time": round(predict_time, 4),
        "accuracy": round(float(score), 4),
        "prediction": prediction,
        "metric_name": metric_name,
    }
