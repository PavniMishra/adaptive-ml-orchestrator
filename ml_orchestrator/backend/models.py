from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Boolean, Text, text
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class Dataset(Base):
    __tablename__ = "datasets"
    dataset_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    size_mb = Column(Float)
    n_features = Column(Integer)
    n_classes = Column(Integer)


class Task(Base):
    __tablename__ = "tasks"
    task_id = Column(Integer, primary_key=True, index=True)
    task_name = Column(String, nullable=False)
    task_type = Column(String, nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.dataset_id"))
    priority = Column(Integer, default=1)
    subtype = Column(String)
    data_fraction = Column(Float, default=1.0)
    status = Column(String, default="pending", index=True)
    algorithm_used = Column(String, default="Priority")
    submit_time = Column(DateTime, default=datetime.utcnow)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)


class TaskPrediction(Base):
    """Input and output for the real ML prediction associated with a scheduled task."""
    __tablename__ = "task_predictions"
    prediction_id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.task_id", ondelete="CASCADE"), unique=True, nullable=False)
    input_data = Column(Text, nullable=False)
    predicted_output = Column(Text, nullable=True)
    evaluation_score = Column(Float, nullable=True)
    metric_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ResourceRequirement(Base):
    __tablename__ = "resource_requirements"
    req_id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.task_id", ondelete="CASCADE"))
    baseline_cpu = Column(Float)
    baseline_ram_mb = Column(Float)
    baseline_duration_sec = Column(Float)
    predicted_cpu = Column(Float)
    predicted_ram_mb = Column(Float)
    predicted_duration_sec = Column(Float)


class ExecutionHistory(Base):
    __tablename__ = "execution_history"
    history_id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.task_id"))
    actual_cpu = Column(Float)
    actual_ram_mb = Column(Float)
    actual_duration_sec = Column(Float)
    wait_time_sec = Column(Float)
    accuracy = Column(Float)
    fit_time_sec = Column(Float)
    predict_time_sec = Column(Float)
    algorithm_used = Column(String)
    is_anomaly = Column(Boolean, default=False)
    execution_time = Column(DateTime, default=datetime.utcnow)


class SchedulingDecision(Base):
    __tablename__ = "scheduling_decisions"
    decision_id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.task_id"))
    decision = Column(String)
    reason = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)


class ResourceLock(Base):
    __tablename__ = "resource_locks"
    lock_id = Column(Integer, primary_key=True, index=True)
    resource_name = Column(String, nullable=False)
    held_by_task_id = Column(Integer, ForeignKey("tasks.task_id"), nullable=True)
    requested_by_task_id = Column(Integer, ForeignKey("tasks.task_id"), nullable=True)


def init_db_triggers(bind_engine):
    """Install the SQLite status-change audit trigger."""
    try:
        with bind_engine.connect() as conn:
            conn.execute(text("""
                CREATE TRIGGER IF NOT EXISTS trg_task_status_audit
                AFTER UPDATE OF status ON tasks
                FOR EACH ROW
                WHEN OLD.status <> NEW.status
                BEGIN
                    INSERT INTO scheduling_decisions (task_id, decision, reason, timestamp)
                    VALUES (NEW.task_id, NEW.status, 'DB_TRIGGER_STATUS_CHANGE', CURRENT_TIMESTAMP);
                END;
            """))
            conn.commit()
    except Exception:
        # Keep startup resilient; the application logs normal scheduling decisions too.
        pass
