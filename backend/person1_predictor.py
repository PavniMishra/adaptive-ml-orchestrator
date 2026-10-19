from sqlalchemy.orm import Session
from sklearn.linear_model import LinearRegression
import numpy as np
import models

MIN_SAMPLES_FOR_MODEL = 5

def predict_advanced(db: Session, task_type: str, dataset_size_mb: float, baseline: dict,
                     n_features: int = None, n_classes: int = None) -> dict:
    """
    Returns {"cpu", "ram", "time", "source"}.
    Multiple Linear Regression on historical runs.
    Features: effective dataset size (MB), number of features, number of classes.
    dataset_size_mb here is the EFFECTIVE size (dataset size x data_fraction).
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

    # clip: no negative values, CPU stays within 0-100
    pred_cpu = min(100.0, max(0.0, model_cpu.predict(X_new)[0]))
    pred_ram = max(0.0, model_ram.predict(X_new)[0])
    pred_time = max(0.0, model_time.predict(X_new)[0])

    return {
        "cpu": round(pred_cpu, 2),
        "ram": round(pred_ram, 2),
        "time": round(pred_time, 2),
        "source": "linear-regression",
    }