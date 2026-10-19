from sqlalchemy.orm import Session
import models

def find_deadlock_pair(db: Session):
    locks = db.query(models.ResourceLock).all()
    waits_on = {}
    for lock in locks:
        if lock.requested_by_task_id and lock.held_by_task_id:
            waits_on.setdefault(lock.requested_by_task_id, set()).add(lock.held_by_task_id)

    for task_a, waiting_for in waits_on.items():
        for task_b in waiting_for:
            if task_a in waits_on.get(task_b, set()):
                return (task_a, task_b)
    return None

def detect_deadlock(db: Session):
    pair = find_deadlock_pair(db)
    return list(pair) if pair else None

def whatif_reduce_ram_ceiling(db: Session, new_ram_ceiling_mb: float):
    """How many past tasks needed more RAM than the new per-task ceiling (in MB)?"""
    history = db.query(models.ExecutionHistory).all()
    blocked = sum(1 for h in history if (h.actual_ram_mb or 0) > new_ram_ceiling_mb)
    max_ram = max((h.actual_ram_mb or 0 for h in history), default=0)
    return {
        "scenario": f"ram_ceiling={new_ram_ceiling_mb} MB",
        "total_tasks": len(history),
        "would_be_blocked": blocked,
        "max_ram_mb": round(max_ram, 2),
    }

def whatif_task_surge(db: Session, extra_tasks: int):
    history = db.query(models.ExecutionHistory).all()
    if history:
        avg_duration = sum(h.actual_duration_sec or 0 for h in history) / len(history)
    else:
        avg_duration = 60.0

    pending_count = db.query(models.Task).filter(models.Task.status == "pending").count()
    projected_extra_wait = (pending_count + extra_tasks) * avg_duration
    return {
        "scenario": f"+{extra_tasks} tasks",
        "projected_extra_wait_sec": round(projected_extra_wait, 2),
    }
