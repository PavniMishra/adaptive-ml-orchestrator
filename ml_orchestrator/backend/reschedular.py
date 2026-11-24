"""
PERSON 2 — Dynamic Rescheduling + Anomaly Detection
======================================================
1. System pressure detection (85% CPU / RAM threshold).
2. Priority-based preemption: pauses lowest-priority running task under pressure.
3. Automatic resumption of paused tasks when load clears.
4. Anomaly detection: flags runs exceeding 1.5x historical usage for the same subtype & size.
"""
from sqlalchemy.orm import Session
import models

ANOMALY_MULTIPLIER = 1.5
PRESSURE_CPU = 85.0
PRESSURE_RAM = 85.0


def average_usage(db: Session, task_type: str, subtype: str = None, data_fraction: float = None):
    q = (
        db.query(models.ExecutionHistory)
        .join(models.Task, models.ExecutionHistory.task_id == models.Task.task_id)
        .filter(models.Task.task_type == task_type)
    )
    if subtype is not None:
        q = q.filter(models.Task.subtype == subtype)
    if data_fraction is not None:
        q = q.filter(models.Task.data_fraction == data_fraction)
    history = q.all()
    if len(history) < 3:
        return None
    avg_cpu = sum(h.actual_cpu or 0 for h in history) / len(history)
    avg_ram = sum(h.actual_ram_mb or 0 for h in history) / len(history)
    return {"avg_cpu": avg_cpu, "avg_ram": avg_ram}


def check_anomaly(db: Session, task_type: str, observed_cpu: float, observed_ram: float,
                  subtype: str = None, data_fraction: float = None) -> bool:
    normal = average_usage(db, task_type, subtype, data_fraction)
    if not normal:
        return False
    return (
        observed_cpu > normal["avg_cpu"] * ANOMALY_MULTIPLIER
        or observed_ram > normal["avg_ram"] * ANOMALY_MULTIPLIER
    )


def detect_pressure(system_cpu_percent: float, system_ram_percent: float) -> bool:
    return system_cpu_percent >= PRESSURE_CPU or system_ram_percent >= PRESSURE_RAM


def pause_lowest_priority_running(db: Session):
    candidate = (
        db.query(models.Task)
        .filter(models.Task.status == "running")
        .order_by(models.Task.priority.asc())
        .first()
    )
    if not candidate:
        return None
    candidate.status = "paused"
    db.add(models.SchedulingDecision(
        task_id=candidate.task_id, decision="paused", reason="resource_pressure_detected",
    ))
    db.commit()
    return candidate.task_id


def resume_paused_tasks(db: Session):
    paused = db.query(models.Task).filter(models.Task.status == "paused").all()
    for t in paused:
        t.status = "pending"
        db.add(models.SchedulingDecision(task_id=t.task_id, decision="resumed", reason="pressure_cleared"))
    if paused:
        db.commit()
    return [t.task_id for t in paused]
