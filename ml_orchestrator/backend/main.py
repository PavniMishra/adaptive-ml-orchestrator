from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
import asyncio
import os
import json
import psutil
from pydantic import BaseModel, Field
import traceback

from database import SessionLocal, engine, get_db
import models
import ml_runner
import person1_predictor as predictor_advanced
import person3_algorithms as algorithms
import person2_rescheduler as rescheduler
import person4_simulator as simulator

models.Base.metadata.create_all(bind=engine)
models.init_db_triggers(engine)
app = FastAPI(title="Adaptive ML Workload Orchestrator — Phase 3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_FRACTIONS = (0.3, 0.6, 1.0)
VALID_OBJECTIVES = ("balanced", "fastest", "priority")
BOOTSTRAP_ROUNDS = 2

# Scheduler state
scheduler_lock = asyncio.Lock()
CURRENT_OBJECTIVE = "balanced"
POLICY_MODE = "recommend"   # "recommend" | "task_type" | "manual"
CURRENT_ALGORITHM = "Priority"
ROUND_ROBIN_QUANTUM = 1.0    # OS Concept: Configurable Time Quantum
POLICY_REASON = "Initial bootstrap state"
CUSTOM_WEIGHTS = {"wait": 0.34, "priority": 0.33, "max_wait": 0.33}
LAST_RR_TASK_ID = None
ACTIVE_EXECUTIONS = set()
SCHEDULER_ENABLED = False  # Scheduling starts only when the user clicks Start Scheduling.


class TaskSubmission(BaseModel):
    task_name: str = Field(min_length=1, max_length=255)
    subtype: str
    data_fraction: float = 1.0
    priority: int = Field(default=1, ge=1, le=10)
    input_data: dict



@app.get("/")
def serve_index():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Frontend not found"}


# ---------- Policy Management ----------

def refresh_policy(db: Session, quantum: float = None):
    """Refreshes active scheduling algorithm based on current policy mode & objective."""
    global CURRENT_ALGORITHM, POLICY_REASON, ROUND_ROBIN_QUANTUM
    if quantum is not None and quantum > 0:
        ROUND_ROBIN_QUANTUM = quantum
    results = algorithms.compare_strategies(db, quantum=ROUND_ROBIN_QUANTUM)

    if POLICY_MODE == "manual":
        POLICY_REASON = f"Manually enforced by user ({CURRENT_ALGORITHM})"
        return results

    if POLICY_MODE == "task_type":
        # Dynamic mapping from recent pending task category
        pending = db.query(models.Task).filter(models.Task.status == "pending").first()
        if pending:
            ds = db.query(models.Dataset).filter(models.Dataset.dataset_id == pending.dataset_id).first()
            if ds:
                eff = ds.size_mb * (pending.data_fraction or 1.0)
                base = algorithms.predict_baseline("Classification", eff)
                pred = predictor_advanced.predict_advanced(db, "Classification", eff, base, ds.n_features, ds.n_classes)
                clf = predictor_advanced.classify_task_type(db, pred["cpu"], pred["ram"], pred["time"], eff, ds.n_features, ds.n_classes)
                CURRENT_ALGORITHM = clf["recommended_algorithm"]
                POLICY_REASON = f"Selected by Task-Type Classification ({clf['category']} -> {CURRENT_ALGORITHM})"
                return results

    # Default: recommendation by objective
    best = algorithms.recommend_strategy(results, CURRENT_OBJECTIVE, CUSTOM_WEIGHTS) if results else None
    if best:
        CURRENT_ALGORITHM = best["algorithm"]
        POLICY_REASON = f"Optimal recommendation for '{CURRENT_OBJECTIVE}' objective"
    else:
        CURRENT_ALGORITHM = "Priority"
        POLICY_REASON = "Default Priority until sufficient execution history exists"
    return results


def policy_payload(db: Session):
    results = refresh_policy(db)
    return {
        "objective": CURRENT_OBJECTIVE,
        "policy_mode": POLICY_MODE,
        "algorithm": CURRENT_ALGORITHM,
        "round_robin_quantum": ROUND_ROBIN_QUANTUM,
        "reason": POLICY_REASON,
        "custom_weights": CUSTOM_WEIGHTS,
        "comparison": results,
    }


# ---------- Scheduler Loop ----------

async def scheduler_loop():
    global LAST_RR_TASK_ID, SCHEDULER_ENABLED
    await asyncio.sleep(1.5)

    while True:
        async with scheduler_lock:
            db = SessionLocal()
            try:
                cpu_percent = psutil.cpu_percent()
                ram_percent = psutil.virtual_memory().percent

                # Resource monitoring remains active. The scheduler only admits
                # pending jobs after the user explicitly starts scheduling.
                if rescheduler.detect_pressure(cpu_percent, ram_percent):
                    rescheduler.pause_lowest_priority_running(db)
                else:
                    rescheduler.resume_paused_tasks(db)

                if SCHEDULER_ENABLED:
                    # OS Starvation Prevention (Aging)
                    algorithms.apply_aging(db, threshold_sec=30.0)

                    # Re-evaluate policy and order the pending queue
                    refresh_policy(db)
                    pending = db.query(models.Task).filter(
                        models.Task.status == "pending"
                    ).all()

                    ordered = []
                    if pending:
                        if CURRENT_ALGORITHM in ("SJF", "SRTF"):
                            ordered = algorithms.order_sjf(pending, db)
                        elif CURRENT_ALGORITHM == "RoundRobin":
                            ordered = algorithms.order_round_robin(
                                pending, LAST_RR_TASK_ID
                            )
                        elif CURRENT_ALGORITHM in ("Priority", "Priority-P"):
                            ordered = algorithms.order_priority(pending)
                        else:
                            ordered = algorithms.order_fcfs(pending)

                    # Admit one task at a time, after the deadlock safety check
                    next_task = ordered[0] if ordered else None
                    if (next_task and not ACTIVE_EXECUTIONS
                            and cpu_percent < rescheduler.PRESSURE_CPU
                            and ram_percent < rescheduler.PRESSURE_RAM):
                        cycle = simulator.detect_deadlock(db)
                        if cycle and next_task.task_id in cycle:
                            db.add(models.SchedulingDecision(
                                task_id=next_task.task_id,
                                decision="rejected",
                                reason=f"deadlock_risk:{cycle}"
                            ))
                            db.commit()
                        else:
                            next_task.status = "running"
                            if next_task.start_time is None:
                                next_task.start_time = datetime.utcnow()
                            next_task.algorithm_used = CURRENT_ALGORITHM
                            LAST_RR_TASK_ID = next_task.task_id
                            db.add(models.SchedulingDecision(
                                task_id=next_task.task_id,
                                decision="admitted",
                                reason=f"{CURRENT_ALGORITHM} ({POLICY_REASON})"
                            ))
                            db.commit()
                            ACTIVE_EXECUTIONS.add(next_task.task_id)
                            asyncio.create_task(run_real_task(next_task.task_id))

                    # Automatically return to idle once the submitted queue drains.
                    elif not pending and not ACTIVE_EXECUTIONS:
                        running_count = db.query(models.Task).filter(
                            models.Task.status == "running"
                        ).count()
                        if running_count == 0:
                            SCHEDULER_ENABLED = False

                else:
                    # Keep the policy display current without executing pending tasks.
                    refresh_policy(db)

            except Exception as e:
                db.rollback()
                print("\\n❌ SCHEDULER ERROR:", repr(e))
                traceback.print_exc()
            finally:
                db.close()

        await asyncio.sleep(2)


async def run_real_task(task_id: int):
    """Run the selected real-data ML workload inside the scheduler worker."""
    try:
        db = SessionLocal()
        try:
            task = db.query(models.Task).filter(models.Task.task_id == task_id).first()
            if not task:
                return
            subtype = task.subtype or "multiclass"
            fraction = task.data_fraction or 1.0
            prediction_row = db.query(models.TaskPrediction).filter(
                models.TaskPrediction.task_id == task_id
            ).first()
            if not prediction_row:
                raise RuntimeError("Task input record was not found.")
            user_input = json.loads(prediction_row.input_data)
        finally:
            db.close()

        result = await asyncio.to_thread(
            ml_runner.run_classification, subtype, fraction, user_input
        )

        db = SessionLocal()
        try:
            task = db.query(models.Task).filter(models.Task.task_id == task_id).first()
            prediction_row = db.query(models.TaskPrediction).filter(
                models.TaskPrediction.task_id == task_id
            ).first()
            if not task or not prediction_row:
                return
            task.status = "completed"
            task.end_time = datetime.utcnow()
            prediction_row.predicted_output = json.dumps(result["prediction"], ensure_ascii=False)
            prediction_row.evaluation_score = result["accuracy"]
            prediction_row.metric_name = result["metric_name"]

            wait_time = (task.start_time - task.submit_time).total_seconds() if task.start_time else 0
            is_anomaly = rescheduler.check_anomaly(
                db, task.task_type, result["cpu"], result["ram_mb"],
                subtype=task.subtype, data_fraction=task.data_fraction)
            db.add(models.ExecutionHistory(
                task_id=task.task_id,
                actual_cpu=result["cpu"], actual_ram_mb=result["ram_mb"],
                actual_duration_sec=result["duration"], wait_time_sec=wait_time,
                accuracy=result["accuracy"], fit_time_sec=result["fit_time"],
                predict_time_sec=result["predict_time"],
                algorithm_used=task.algorithm_used, is_anomaly=is_anomaly))
            db.commit()
            refresh_policy(db)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
    except Exception as exc:
        db = SessionLocal()
        try:
            task = db.query(models.Task).filter(models.Task.task_id == task_id).first()
            prediction_row = db.query(models.TaskPrediction).filter(
                models.TaskPrediction.task_id == task_id
            ).first()
            if task:
                task.status = "failed"
                task.end_time = datetime.utcnow()
            if prediction_row:
                prediction_row.predicted_output = "ERROR: " + str(exc)
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()
    finally:
        ACTIVE_EXECUTIONS.discard(task_id)


# ---------- Task Creation ----------

def _create_task(db: Session, task_name: str, subtype: str, data_fraction: float, priority: int):
    dataset_name = ml_runner.SUBTYPE_TO_DATASET.get(subtype, ml_runner.SUBTYPE_TO_DATASET["multiclass"])
    dataset = db.query(models.Dataset).filter(models.Dataset.name == dataset_name).first()
    if not dataset:
        ml_runner.ensure_datasets(db)
        dataset = db.query(models.Dataset).filter(models.Dataset.name == dataset_name).first()

    new_task = models.Task(
        task_name=task_name, task_type="Classification", dataset_id=dataset.dataset_id,
        priority=priority, subtype=subtype, data_fraction=data_fraction,
    )
    db.add(new_task)
    db.flush()

    effective_size = dataset.size_mb * data_fraction
    baseline = algorithms.predict_baseline("Classification", effective_size)
    advanced = predictor_advanced.predict_advanced(
        db, "Classification", effective_size, baseline,
        n_features=dataset.n_features, n_classes=dataset.n_classes)

    db.add(models.ResourceRequirement(
        task_id=new_task.task_id,
        baseline_cpu=baseline["cpu"], baseline_ram_mb=baseline["ram"], baseline_duration_sec=baseline["time"],
        predicted_cpu=advanced["cpu"], predicted_ram_mb=advanced["ram"], predicted_duration_sec=advanced["time"],
    ))

    # Logistic Regression task classification
    classification = predictor_advanced.classify_task_type(
        db, advanced["cpu"], advanced["ram"], advanced["time"],
        effective_size, dataset.n_features, dataset.n_classes
    )

    return new_task, advanced, classification


def bootstrap_history(db: Session):
    """Do not seed fabricated ML scores; real execution history starts with real jobs."""
    return
    if db.query(models.Task).count() > 0:
        return

    # Seed an initial completed benchmark pass so Linear Regression & Prediction Accuracy
    # have historical data immediately instead of showing 0s until 5 live jobs finish.
    seed_profiles = {
        "binary":     {"cpu_base": 14.0, "ram_per_mb": 11.5, "time_per_mb": 0.45, "acc": 0.965},
        "multiclass": {"cpu_base": 18.5, "ram_per_mb": 13.0, "time_per_mb": 0.85, "acc": 0.912},
        "multilabel": {"cpu_base": 24.0, "ram_per_mb": 14.5, "time_per_mb": 1.25, "acc": 0.845},
    }

    now = datetime.utcnow()
    idx = 0
    for subtype in ml_runner.SUBTYPE_TO_DATASET:
        prof = seed_profiles[subtype]
        for fraction in VALID_FRACTIONS:
            t_seed, _, _ = _create_task(
                db,
                f"Seed-{subtype}-{int(fraction * 100)}%",
                subtype,
                fraction,
                algorithms.BATCH_PRIORITIES[idx % len(algorithms.BATCH_PRIORITIES)],
            )
            db.flush()
            ds = db.query(models.Dataset).filter(models.Dataset.dataset_id == t_seed.dataset_id).first()
            eff = (ds.size_mb if ds else 10.0) * fraction
            actual_cpu = round(min(95.0, prof["cpu_base"] + eff * 0.15), 2)
            actual_ram = round(max(20.0, eff * prof["ram_per_mb"]), 2)
            actual_dur = round(max(0.2, eff * prof["time_per_mb"]), 4)
            t_seed.status = "completed"
            t_seed.algorithm_used = "Priority"
            t_seed.start_time = now
            t_seed.end_time = now
            db.add(models.ExecutionHistory(
                task_id=t_seed.task_id,
                actual_cpu=actual_cpu,
                actual_ram_mb=actual_ram,
                actual_duration_sec=actual_dur,
                wait_time_sec=round(0.3 * (idx + 1), 2),
                accuracy=prof["acc"],
                fit_time_sec=round(actual_dur * 0.8, 4),
                predict_time_sec=round(actual_dur * 0.2, 4),
                algorithm_used="Priority",
                is_anomaly=False,
                execution_time=now,
            ))
            idx += 1

    # Also create regression-evaluated benchmark tasks so MAE & points are populated on load
    for _ in range(BOOTSTRAP_ROUNDS):
        i = 0
        for subtype in ml_runner.SUBTYPE_TO_DATASET:
            prof = seed_profiles[subtype]
            for fraction in VALID_FRACTIONS:
                t_bench, adv, _ = _create_task(
                    db,
                    f"Benchmark-{subtype}-{int(fraction * 100)}%",
                    subtype,
                    fraction,
                    algorithms.BATCH_PRIORITIES[i % len(algorithms.BATCH_PRIORITIES)],
                )
                db.flush()
                if adv.get("source") == "linear-regression":
                    ds = db.query(models.Dataset).filter(models.Dataset.dataset_id == t_bench.dataset_id).first()
                    eff = (ds.size_mb if ds else 10.0) * fraction
                    t_bench.status = "completed"
                    t_bench.algorithm_used = "Priority"
                    t_bench.start_time = now
                    t_bench.end_time = now
                    db.add(models.ExecutionHistory(
                        task_id=t_bench.task_id,
                        actual_cpu=round(min(95.0, prof["cpu_base"] + eff * 0.16), 2),
                        actual_ram_mb=round(max(20.0, eff * prof["ram_per_mb"] * 1.03), 2),
                        actual_duration_sec=round(max(0.2, eff * prof["time_per_mb"] * 0.96), 4),
                        wait_time_sec=round(0.4 * (i + 1), 2),
                        accuracy=prof["acc"],
                        fit_time_sec=round(eff * prof["time_per_mb"] * 0.77, 4),
                        predict_time_sec=round(eff * prof["time_per_mb"] * 0.19, 4),
                        algorithm_used="Priority",
                        is_anomaly=False,
                        execution_time=now,
                    ))
                i += 1
    db.commit()


@app.on_event("startup")
async def startup_event():
    db = SessionLocal()
    try:
        models.init_db_triggers(engine)
        ml_runner.ensure_datasets(db)
        bootstrap_history(db)
        refresh_policy(db)
    finally:
        db.close()
    asyncio.create_task(scheduler_loop())


# ---------- REST Endpoints ----------

@app.post("/submit_task/")
def submit_task(payload: TaskSubmission, db: Session = Depends(get_db)):
    if payload.subtype not in ml_runner.SUBTYPE_TO_DATASET:
        raise HTTPException(status_code=400, detail="Invalid classification subtype.")
    if payload.data_fraction not in VALID_FRACTIONS:
        raise HTTPException(status_code=400, detail="Data fraction must be 0.3, 0.6, or 1.0.")
    if not payload.task_name.strip():
        raise HTTPException(status_code=400, detail="Task name is required.")
    try:
        new_task, advanced, task_clf = _create_task(
            db, payload.task_name.strip(), payload.subtype,
            payload.data_fraction, payload.priority)
        db.add(models.TaskPrediction(
            task_id=new_task.task_id,
            input_data=json.dumps(payload.input_data, ensure_ascii=False)))
        db.commit()
        return {
            "message": "Task submitted successfully",
            "task_id": new_task.task_id,
            "status": new_task.status,
            "prediction_source": advanced["source"],
            "predictions": {"cpu": advanced["cpu"], "ram_mb": advanced["ram"],
                            "duration_sec": advanced["time"]},
            "task_classification": task_clf,
        }
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/task_result/{task_id}")
def task_result(task_id: int, db: Session = Depends(get_db)):
    task = db.query(models.Task).filter(models.Task.task_id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    row = db.query(models.TaskPrediction).filter(
        models.TaskPrediction.task_id == task_id).first()
    output = row.predicted_output if row else None
    if output and output.startswith("ERROR: "):
        return {"task_id": task_id, "status": task.status, "error": output[7:]}
    try:
        prediction = json.loads(output) if output else None
    except (TypeError, json.JSONDecodeError):
        prediction = output
    return {
        "task_id": task_id,
        "status": task.status,
        "prediction": prediction,
        "score": row.evaluation_score if row else None,
        "metric_name": row.metric_name if row else None,
    }


@app.get("/scheduler_status/")
def scheduler_status(db: Session = Depends(get_db)):
    return {
        "enabled": SCHEDULER_ENABLED,
        "pending": db.query(models.Task).filter(models.Task.status == "pending").count(),
        "running": db.query(models.Task).filter(models.Task.status == "running").count(),
        "completed": db.query(models.Task).filter(models.Task.status == "completed").count(),
        "failed": db.query(models.Task).filter(models.Task.status == "failed").count(),
    }


@app.post("/start_scheduling/")
async def start_scheduling(db: Session = Depends(get_db)):
    global SCHEDULER_ENABLED

    async with scheduler_lock:
        pending_count = db.query(models.Task).filter(
            models.Task.status == "pending"
        ).count()
        running_count = db.query(models.Task).filter(
            models.Task.status == "running"
        ).count()

        if pending_count == 0 and running_count == 0 and not ACTIVE_EXECUTIONS:
            return {
                "message": "There are no pending tasks. Submit tasks before starting the scheduler.",
                "enabled": False,
                "pending": 0,
            }

        SCHEDULER_ENABLED = True

    return {
        "message": "Scheduling started. Pending tasks will be processed using the active scheduling policy.",
        "enabled": True,
        "pending": pending_count,
    }


@app.get("/task_queue/")
def task_queue(db: Session = Depends(get_db)):
    tasks = db.query(models.Task).order_by(
        models.Task.submit_time.asc(), models.Task.task_id.asc()
    ).all()

    result = []
    for task in tasks:
        prediction_row = db.query(models.TaskPrediction).filter(
            models.TaskPrediction.task_id == task.task_id
        ).first()

        prediction = None
        if prediction_row and prediction_row.predicted_output:
            raw_prediction = prediction_row.predicted_output
            try:
                prediction = json.loads(raw_prediction)
            except (TypeError, json.JSONDecodeError):
                prediction = raw_prediction

        result.append({
            "task_id": task.task_id,
            "task_name": task.task_name,
            "subtype": task.subtype,
            "priority": task.priority,
            "status": task.status,
            "algorithm_used": task.algorithm_used,
            "prediction": prediction,
        })

    return result


@app.get("/datasets/")
def get_datasets(db: Session = Depends(get_db)):
    ml_runner.ensure_datasets(db)
    rows = db.query(models.Dataset).all()
    return [{
        "dataset_id": r.dataset_id,
        "name": r.name,
        "size_mb": r.size_mb,
        "n_features": r.n_features,
        "n_classes": r.n_classes,
    } for r in rows]


# ---------- Objective, Policy & Strategy ----------

@app.post("/objective/")
def set_objective(
    objective: str = "balanced",
    weight_wait: float = None,
    weight_priority: float = None,
    weight_max: float = None,
    db: Session = Depends(get_db)
):
    global CURRENT_OBJECTIVE, POLICY_MODE, CUSTOM_WEIGHTS
    if objective not in VALID_OBJECTIVES:
        raise HTTPException(status_code=400, detail=f"objective must be one of {list(VALID_OBJECTIVES)}")
    CURRENT_OBJECTIVE = objective
    POLICY_MODE = "recommend"

    if weight_wait is not None or weight_priority is not None or weight_max is not None:
        CUSTOM_WEIGHTS = {
            "wait": weight_wait if weight_wait is not None else CUSTOM_WEIGHTS["wait"],
            "priority": weight_priority if weight_priority is not None else CUSTOM_WEIGHTS["priority"],
            "max_wait": weight_max if weight_max is not None else CUSTOM_WEIGHTS["max_wait"],
        }

    return policy_payload(db)


@app.post("/policy_mode/")
def set_policy_mode(mode: str, algorithm: str = None, quantum: float = None, db: Session = Depends(get_db)):
    """Allows choosing between 'recommend' (objective-based), 'task_type' (logistic regression), or 'manual'."""
    global POLICY_MODE, CURRENT_ALGORITHM, ROUND_ROBIN_QUANTUM
    if mode not in ("recommend", "task_type", "manual"):
        raise HTTPException(status_code=400, detail="mode must be 'recommend', 'task_type', or 'manual'")
    POLICY_MODE = mode
    if mode == "manual" and algorithm:
        CURRENT_ALGORITHM = algorithm
    if quantum is not None and quantum > 0:
        ROUND_ROBIN_QUANTUM = quantum
    return policy_payload(db)


@app.post("/strategy/")
def set_strategy(algorithm: str, db: Session = Depends(get_db)):
    """Backward compatibility endpoint for manual strategy selection."""
    global POLICY_MODE, CURRENT_ALGORITHM
    POLICY_MODE = "manual"
    CURRENT_ALGORITHM = algorithm
    return policy_payload(db)


@app.get("/strategy_comparison/")
def strategy_comparison(objective: str = "balanced", quantum: float = 1.0, db: Session = Depends(get_db)):
    results = algorithms.compare_strategies(db, quantum=quantum)
    recommended = algorithms.recommend_strategy(results, objective, CUSTOM_WEIGHTS)
    return {"all_strategies": results, "recommended": recommended, "quantum": quantum}


@app.post("/apply_aging/")
def trigger_aging(threshold_sec: float = 30.0, db: Session = Depends(get_db)):
    """OS Concept: Starvation prevention via Priority Aging."""
    aged = algorithms.apply_aging(db, threshold_sec=threshold_sec)
    return {"message": f"Aged {len(aged)} tasks waiting >= {threshold_sec}s", "aged_task_ids": aged}


@app.get("/current_policy/")
def current_policy(db: Session = Depends(get_db)):
    return policy_payload(db)


# ---------- Task Classification (Logistic Regression) ----------

@app.get("/classify_task/")
def classify_task(subtype: str = "binary", data_fraction: float = 1.0, db: Session = Depends(get_db)):
    ds_name = ml_runner.SUBTYPE_TO_DATASET.get(subtype, "Synthetic Binary")
    ds = db.query(models.Dataset).filter(models.Dataset.name == ds_name).first()
    if not ds:
        ml_runner.ensure_datasets(db)
        ds = db.query(models.Dataset).filter(models.Dataset.name == ds_name).first()

    eff = ds.size_mb * data_fraction
    base = algorithms.predict_baseline("Classification", eff)
    pred = predictor_advanced.predict_advanced(db, "Classification", eff, base, ds.n_features, ds.n_classes)
    return predictor_advanced.classify_task_type(
        db, pred["cpu"], pred["ram"], pred["time"], eff, ds.n_features, ds.n_classes
    )


# ---------- Prediction Accuracy Analytics ----------

def _regression_rows(db: Session):
    rows = (
        db.query(models.ResourceRequirement, models.ExecutionHistory, models.Task)
        .join(models.ExecutionHistory, models.ExecutionHistory.task_id == models.ResourceRequirement.task_id)
        .join(models.Task, models.Task.task_id == models.ResourceRequirement.task_id)
        .all()
    )
    used = [
        (r, h, t) for r, h, t in rows
        if r.predicted_ram_mb is not None and r.baseline_ram_mb is not None
        and abs(r.predicted_ram_mb - r.baseline_ram_mb) > 1e-6
    ]
    return rows, used


@app.get("/prediction_accuracy/")
def prediction_accuracy(db: Session = Depends(get_db)):
    rows, used = _regression_rows(db)

    def stats(pred_attr, actual_attr):
        vals = [(getattr(r, pred_attr), getattr(h, actual_attr)) for r, h, _ in used]
        vals = [(p, a) for p, a in vals if p is not None and a is not None and a > 0]
        if not vals:
            return {"mae": 0.0, "mape_percent": 0.0}
        mae = sum(abs(p - a) for p, a in vals) / len(vals)
        mape = sum(abs(p - a) / a for p, a in vals) / len(vals) * 100
        return {"mae": round(mae, 2), "mape_percent": round(mape, 1)}

    return {
        "total_tasks": len(rows),
        "regression_tasks": len(used),
        "ram_mb": {
            "baseline": stats("baseline_ram_mb", "actual_ram_mb"),
            "regression": stats("predicted_ram_mb", "actual_ram_mb"),
        },
        "duration_sec": {
            "baseline": stats("baseline_duration_sec", "actual_duration_sec"),
            "regression": stats("predicted_duration_sec", "actual_duration_sec"),
        },
    }


@app.get("/prediction_points/")
def prediction_points(db: Session = Depends(get_db)):
    _, used = _regression_rows(db)
    return [{
        "subtype": t.subtype,
        "fraction": t.data_fraction,
        "pred_ram": r.predicted_ram_mb, "actual_ram": h.actual_ram_mb,
        "pred_time": r.predicted_duration_sec, "actual_time": h.actual_duration_sec,
    } for r, h, t in used]


# ---------- Deadlock Avoidance (Banker's Algorithm & What-If) ----------

@app.get("/bankers/demo/")
def bankers_demo(p: int = None, r_a: int = None, r_b: int = None, r_c: int = None):
    custom_vec = [r_a, r_b, r_c] if (r_a is not None and r_b is not None and r_c is not None) else None
    return simulator.run_bankers_algorithm(custom_request_p=p, custom_request_vec=custom_vec)


@app.get("/deadlock_check/")
def deadlock_check(db: Session = Depends(get_db)):
    cycle = simulator.detect_deadlock(db)
    return {"deadlock_detected": cycle is not None, "cycle": cycle}


@app.get("/whatif/ram/")
def whatif_ram(new_ceiling_mb: float = None, new_ceiling_percent: float = None, db: Session = Depends(get_db)):
    # Support both MB input and legacy percent input gracefully
    if new_ceiling_mb is None and new_ceiling_percent is not None:
        new_ceiling_mb = (new_ceiling_percent / 100.0) * 1024.0
    if new_ceiling_mb is None:
        new_ceiling_mb = 150.0
    return simulator.whatif_reduce_ram_ceiling(db, new_ceiling_mb)


@app.get("/whatif/surge/")
def whatif_surge(extra_tasks: int = 5, db: Session = Depends(get_db)):
    return simulator.whatif_task_surge(db, extra_tasks)


# ---------- Explain-a-Task & Audit Log ----------

@app.get("/explain_task/{task_id}")
def explain_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(models.Task).filter(models.Task.task_id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    ds = db.query(models.Dataset).filter(models.Dataset.dataset_id == task.dataset_id).first()
    req = db.query(models.ResourceRequirement).filter(models.ResourceRequirement.task_id == task_id).first()
    hist = db.query(models.ExecutionHistory).filter(models.ExecutionHistory.task_id == task_id).first()
    decision = (
        db.query(models.SchedulingDecision)
        .filter(models.SchedulingDecision.task_id == task_id)
        .order_by(models.SchedulingDecision.timestamp.desc())
        .first()
    )

    prediction_row = db.query(models.TaskPrediction).filter(
        models.TaskPrediction.task_id == task_id).first()
    try:
        saved_prediction = json.loads(prediction_row.predicted_output) if prediction_row and prediction_row.predicted_output and not prediction_row.predicted_output.startswith("ERROR: ") else None
    except (TypeError, json.JSONDecodeError):
        saved_prediction = prediction_row.predicted_output if prediction_row else None

    clf = None
    if req and ds:
        clf = predictor_advanced.classify_task_type(
            db, req.predicted_cpu or 10.0, req.predicted_ram_mb or 100.0,
            req.predicted_duration_sec or 1.0, ds.size_mb * (task.data_fraction or 1.0),
            ds.n_features, ds.n_classes
        )

    return {
        "task": {
            "task_id": task.task_id,
            "name": task.task_name,
            "subtype": task.subtype,
            "data_fraction": task.data_fraction,
            "priority": task.priority,
            "status": task.status,
            "algorithm_used": task.algorithm_used,
            "submit_time": task.submit_time,
            "start_time": task.start_time,
            "end_time": task.end_time,
        },
        "dataset": {
            "name": ds.name if ds else "Unknown",
            "size_mb": ds.size_mb if ds else None,
            "n_features": ds.n_features if ds else None,
            "n_classes": ds.n_classes if ds else None,
            "effective_size_mb": (ds.size_mb * (task.data_fraction or 1.0)) if ds else None,
        },
        "predictions": {
            "baseline": {
                "cpu": req.baseline_cpu if req else None,
                "ram_mb": req.baseline_ram_mb if req else None,
                "duration_sec": req.baseline_duration_sec if req else None,
            },
            "linear_regression": {
                "cpu": req.predicted_cpu if req else None,
                "ram_mb": req.predicted_ram_mb if req else None,
                "duration_sec": req.predicted_duration_sec if req else None,
            },
            "classification": clf,
        },
        "scheduling_decision": {
            "decision": decision.decision if decision else "Pending",
            "reason": decision.reason if decision else "Waiting in queue",
            "timestamp": decision.timestamp if decision else None,
        },
        "ml_prediction": {
            "input": json.loads(prediction_row.input_data) if prediction_row else None,
            "prediction": saved_prediction,
            "score": prediction_row.evaluation_score if prediction_row else None,
            "metric_name": prediction_row.metric_name if prediction_row else None,
        } if prediction_row else None,
        "actual_execution": {
            "cpu_percent": hist.actual_cpu if hist else None,
            "ram_mb": hist.actual_ram_mb if hist else None,
            "duration_sec": hist.actual_duration_sec if hist else None,
            "wait_time_sec": hist.wait_time_sec if hist else None,
            "accuracy": hist.accuracy if hist else None,
            "is_anomaly": hist.is_anomaly if hist else False,
        } if hist else None,
    }


@app.get("/analytics/")
def get_analytics(db: Session = Depends(get_db)):
    query = text("""
        SELECT t.subtype, AVG(e.wait_time_sec) as avg_wait,
               SUM(CASE WHEN e.is_anomaly THEN 1 ELSE 0 END) as anomalies,
               AVG(e.accuracy) as avg_accuracy
        FROM execution_history e
        JOIN tasks t ON e.task_id = t.task_id
        GROUP BY t.subtype
        ORDER BY t.subtype
    """)
    result = db.execute(query).fetchall()
    return [{"task_type": r[0] or "General", "avg_wait": round(r[1] or 0.0, 2), "anomalies": r[2] or 0, "avg_accuracy": round(r[3] or 0.0, 3)} for r in result]


@app.get("/scheduling_decisions/")
def scheduling_decisions(db: Session = Depends(get_db)):
    rows = (
        db.query(models.SchedulingDecision)
        .order_by(models.SchedulingDecision.timestamp.desc(), models.SchedulingDecision.decision_id.desc())
        .limit(50)
        .all()
    )
    return [{"task_id": r.task_id, "decision": r.decision, "reason": r.reason, "timestamp": r.timestamp} for r in rows]


@app.post("/setup/")
def setup_default(db: Session = Depends(get_db)):
    ml_runner.ensure_datasets(db)
    bootstrap_history(db)
    refresh_policy(db)
    return {"message": "Phase 3 datasets and benchmark history initialized successfully."}
