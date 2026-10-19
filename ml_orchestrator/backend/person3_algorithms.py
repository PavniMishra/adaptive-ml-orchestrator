from sqlalchemy.orm import Session
import models
import person1_predictor as predictor

HIGH_PRIORITY = 7   # priority 7 or more counts as high priority
SUBTYPES = ["binary", "multiclass", "multilabel"]
FRACTIONS = [0.3, 0.6, 1.0]
# fixed priorities of the standard 9-task batch, mixed so that priority is not tied to task size
BATCH_PRIORITIES = [9, 3, 5, 1, 8, 7, 6, 2, 4]

def predict_baseline(task_type: str, dataset_size_mb: float) -> dict:
    base_cpu = 10.0
    base_ram = 50.0
    if task_type == "Training":
        return {"cpu": base_cpu * 2, "ram": base_ram * dataset_size_mb / 100, "time": dataset_size_mb / 50 * 2}
    if task_type == "Image Processing":
        return {"cpu": base_cpu * 1.5, "ram": base_ram * dataset_size_mb / 150, "time": dataset_size_mb / 50 * 1.5}
    return {"cpu": base_cpu, "ram": base_ram * dataset_size_mb / 200, "time": dataset_size_mb / 50}

def order_fcfs(tasks):
    return sorted(tasks, key=lambda t: t.submit_time)

def order_sjf(tasks, db: Session):
    def predicted_duration(t):
        req = db.query(models.ResourceRequirement).filter(models.ResourceRequirement.task_id == t.task_id).first()
        return req.predicted_duration_sec if req and req.predicted_duration_sec else 9999
    return sorted(tasks, key=predicted_duration)

def order_priority(tasks):
    return sorted(tasks, key=lambda t: -t.priority)

def order_round_robin(tasks, last_served_id=None):
    if not tasks:
        return tasks
    ids = [t.task_id for t in tasks]
    if last_served_id in ids:
        idx = ids.index(last_served_id)
        return tasks[idx + 1:] + tasks[: idx + 1]
    return tasks

ALGORITHMS = {
    "FCFS": order_fcfs,
    "SJF": order_sjf,
    "Priority": order_priority,
    "RoundRobin": order_round_robin,
}

def _standard_batch(db: Session):
    """
    Standard 9-task batch (3 subtypes x 3 sizes), all arriving together.
    actual    = average measured duration of that subtype+size from execution_history
    predicted = Linear Regression prediction (used by SJF to order tasks)
    Returns None if some combination has no run history yet.
    """
    batch = []
    i = 0
    for subtype in SUBTYPES:
        ds = (
            db.query(models.Dataset)
            .join(models.Task, models.Task.dataset_id == models.Dataset.dataset_id)
            .filter(models.Task.subtype == subtype)
            .first()
        )
        for fraction in FRACTIONS:
            runs = (
                db.query(models.ExecutionHistory.actual_duration_sec)
                .join(models.Task, models.ExecutionHistory.task_id == models.Task.task_id)
                .filter(models.Task.subtype == subtype, models.Task.data_fraction == fraction)
                .all()
            )
            durations = [r[0] for r in runs if r[0] is not None]
            if not ds or not durations:
                return None
            effective_size = ds.size_mb * fraction
            baseline = predict_baseline("Classification", effective_size)
            pred = predictor.predict_advanced(
                db, "Classification", effective_size, baseline,
                n_features=ds.n_features, n_classes=ds.n_classes)
            batch.append({
                "actual": sum(durations) / len(durations),
                "predicted": pred["time"],
                "priority": BATCH_PRIORITIES[i],
            })
            i += 1
    return batch

def _simulate(order):
    """Runs the tasks one after another in the given order and measures waiting times."""
    clock = 0.0
    waits = []
    for job in order:
        waits.append((job, clock))   # a task waits until all tasks before it have finished
        clock += job["actual"]
    all_waits = [w for _, w in waits]
    high = [w for j, w in waits if j["priority"] >= HIGH_PRIORITY]
    return {
        "avg_wait_sec": round(sum(all_waits) / len(all_waits), 2),
        "high_priority_wait_sec": round(sum(high) / len(high), 2) if high else None,
        "max_wait_sec": round(max(all_waits), 2),
    }

def compare_strategies(db: Session):
    """Compares all 4 algorithms on the same standard batch. Returns [] if there is no history yet."""
    batch = _standard_batch(db)
    if not batch:
        return []
    orders = {
        "FCFS": list(batch),
        "SJF": sorted(batch, key=lambda j: j["predicted"]),
        "Priority": sorted(batch, key=lambda j: -j["priority"]),
        # the running scheduler's Round Robin is non-preemptive, so it behaves like FCFS
        "RoundRobin": list(batch),
    }
    return [{"algorithm": algo, **_simulate(order)} for algo, order in orders.items()]

def recommend_strategy(results, objective: str):
    if not results:
        return None
    if objective == "fastest":
        return min(results, key=lambda r: r["avg_wait_sec"])
    if objective == "priority":
        valid = [r for r in results if r["high_priority_wait_sec"] is not None]
        return min(valid, key=lambda r: r["high_priority_wait_sec"]) if valid else None

    # balanced: divide each metric by its worst value, then pick the lowest total
    def norm(key):
        vals = [r[key] for r in results if r[key] is not None]
        worst = max(vals) if vals and max(vals) > 0 else 1
        return lambda r: (r[key] if r[key] is not None else worst) / worst
    scorers = [norm("avg_wait_sec"), norm("high_priority_wait_sec"), norm("max_wait_sec")]
    return min(results, key=lambda r: sum(s(r) for s in scorers))