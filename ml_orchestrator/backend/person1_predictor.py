"""
PERSON 1 — Predictive Allocation & Task Classification (Pavni Mishra)
======================================================================
1. Multiple Linear Regression: predicts CPU, RAM, and execution duration
   based on effective dataset size, features, classes, and fraction.
2. Logistic Regression: classifies workloads into 'CPU-intensive',
   'RAM-intensive', or 'balanced' and maps them to the optimal algorithm.
"""
from sqlalchemy.orm import Session
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import models

MIN_SAMPLES_FOR_MODEL = 5

TASK_TYPE_ALGORITHM_MAP = {
    "CPU-intensive": "SJF",
    "RAM-intensive": "Priority",
    "balanced": "RoundRobin",
}


def predict_advanced(db: Session, task_type: str, dataset_size_mb: float, baseline: dict,
                     n_features: int = None, n_classes: int = None) -> dict:
    """
    Multiple Linear Regression on historical runs.
    Features: effective dataset size (MB), number of features, number of classes.
    """
    records = (
        db.query(models.ExecutionHistory, models.Dataset, models.Task)
        .join(models.Task, models.ExecutionHistory.task_id == models.Task.task_id)
        .join(models.Dataset, models.Task.dataset_id == models.Dataset.dataset_id)
        .filter(models.Task.task_type == task_type)
        .all()
    )

    if len(records) < MIN_SAMPLES_FOR_MODEL or n_features is None or n_classes is None:
        return {**baseline, "source": "baseline-fallback"}

    X = np.array([
        [(d.size_mb or 0) * (t.data_fraction or 1.0), d.n_features or 0, d.n_classes or 0]
        for _, d, t in records
    ])
    y_cpu = np.array([h.actual_cpu or 0 for h, _, _ in records])
    y_ram = np.array([h.actual_ram_mb or 0 for h, _, _ in records])
    y_time = np.array([h.actual_duration_sec or 0 for h, _, _ in records])

    model_cpu = LinearRegression().fit(X, y_cpu)
    model_ram = LinearRegression().fit(X, y_ram)
    model_time = LinearRegression().fit(X, y_time)

    X_new = np.array([[dataset_size_mb, n_features, n_classes]])

    pred_cpu = min(100.0, max(0.0, float(model_cpu.predict(X_new)[0])))
    pred_ram = max(0.0, float(model_ram.predict(X_new)[0]))
    pred_time = max(0.0, float(model_time.predict(X_new)[0]))

    return {
        "cpu": round(pred_cpu, 2),
        "ram": round(pred_ram, 2),
        "time": round(pred_time, 2),
        "source": "linear-regression",
    }


def _determine_label(cpu: float, ram: float) -> str:
    """Helper to label workload profile based on resource footprint."""
    if cpu > 25.0 and ram < 150.0:
        return "CPU-intensive"
    elif ram > 200.0:
        return "RAM-intensive"
    else:
        return "balanced"


def classify_task_type(db: Session, predicted_cpu: float, predicted_ram: float,
                       predicted_time: float, effective_size: float = 10.0,
                       n_features: int = 30, n_classes: int = 2) -> dict:
    """
    Logistic Regression model that classifies tasks into:
    'CPU-intensive', 'RAM-intensive', or 'balanced'.
    Maps the label to the corresponding scheduling algorithm.
    """
    # Seed training data representing standard feature clusters
    seed_X = [
        [40.0, 50.0, 1.2],    # High CPU, low RAM -> CPU-intensive
        [35.0, 70.0, 2.0],
        [38.0, 60.0, 1.5],
        [10.0, 350.0, 5.0],   # Low CPU, high RAM -> RAM-intensive
        [12.0, 420.0, 6.2],
        [15.0, 300.0, 4.5],
        [20.0, 120.0, 2.5],   # Moderate CPU, moderate RAM -> balanced
        [18.0, 140.0, 3.0],
        [22.0, 110.0, 2.1],
    ]
    seed_y = [
        "CPU-intensive", "CPU-intensive", "CPU-intensive",
        "RAM-intensive", "RAM-intensive", "RAM-intensive",
        "balanced", "balanced", "balanced",
    ]

    # Incorporate historical runs if available
    records = db.query(models.ExecutionHistory).all() if db else []
    for h in records:
        if h.actual_cpu and h.actual_ram_mb:
            seed_X.append([h.actual_cpu, h.actual_ram_mb, h.actual_duration_sec or 1.0])
            seed_y.append(_determine_label(h.actual_cpu, h.actual_ram_mb))

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(seed_X, seed_y)

    query_point = np.array([[predicted_cpu, predicted_ram, predicted_time]])
    category = str(clf.predict(query_point)[0])
    probs = dict(zip(clf.classes_, [round(float(p), 3) for p in clf.predict_proba(query_point)[0]]))

    recommended_algo = TASK_TYPE_ALGORITHM_MAP.get(category, "RoundRobin")

    return {
        "category": category,
        "recommended_algorithm": recommended_algo,
        "probabilities": probs,
        "features": {
            "predicted_cpu": predicted_cpu,
            "predicted_ram": predicted_ram,
            "predicted_time": predicted_time,
        },
    }
