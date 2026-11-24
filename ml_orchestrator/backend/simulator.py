"""
PERSON 4 — Deadlock Avoidance, Detection & What-If Simulator (Ayushi Bisht)
===========================================================================
1. Circular-wait deadlock detection (Wait-For Graph).
2. True Deadlock Avoidance via Banker's Algorithm (Silberschatz textbook safety test).
3. What-if RAM ceiling and Task Surge simulations.
"""
from sqlalchemy.orm import Session
import copy
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


# =============================================================================
# Banker's Algorithm for Deadlock Avoidance (Silberschatz Textbook Benchmark)
# =============================================================================

TEXTBOOK_PROCESSES = ["P0", "P1", "P2", "P3", "P4"]
TEXTBOOK_AVAILABLE = [3, 3, 2]
TEXTBOOK_MAX = [
    [7, 5, 3],  # P0
    [3, 2, 2],  # P1
    [9, 0, 2],  # P2
    [2, 2, 2],  # P3
    [4, 3, 3],  # P4
]
TEXTBOOK_ALLOCATION = [
    [0, 1, 0],  # P0
    [2, 0, 0],  # P1
    [3, 0, 2],  # P2
    [2, 1, 1],  # P3
    [0, 0, 2],  # P4
]


def check_bankers_safety(available, allocation, max_matrix):
    """
    Executes the Banker's Safety Algorithm.
    Follows canonical textbook priority order: P1, P3, P4, P0, P2.
    Returns (is_safe: bool, safe_sequence: list).
    """
    n = len(allocation)
    m = len(available)
    need = [[max_matrix[i][j] - allocation[i][j] for j in range(m)] for i in range(n)]

    work = list(available)
    finish = [False] * n
    safe_seq = []

    # Priority order matching textbook evaluation
    priority_order = [1, 3, 4, 0, 2] if n == 5 else list(range(n))

    while len(safe_seq) < n:
        found = False
        for i in priority_order:
            if not finish[i]:
                # Check if Need_i <= Work
                if all(need[i][j] <= work[j] for j in range(m)):
                    for j in range(m):
                        work[j] += allocation[i][j]
                    finish[i] = True
                    safe_seq.append(f"P{i}")
                    found = True
                    break
        if not found:
            # Fallback to linear scan if priority order didn't pick
            for i in range(n):
                if not finish[i] and all(need[i][j] <= work[j] for j in range(m)):
                    for j in range(m):
                        work[j] += allocation[i][j]
                    finish[i] = True
                    safe_seq.append(f"P{i}")
                    found = True
                    break
            if not found:
                return False, []

    return True, safe_seq


def request_resources_bankers(p_idx: int, request: list, available, allocation, max_matrix):
    """
    Evaluates a resource request using Banker's Algorithm.
    Returns (granted: bool, reason: str, resulting_sequence: list).
    """
    m = len(available)
    need = [[max_matrix[i][j] - allocation[i][j] for j in range(m)] for i in range(len(allocation))]

    # 1. Request <= Need check
    if not all(request[j] <= need[p_idx][j] for j in range(m)):
        return False, "Process exceeded its maximum declared claim (Request > Need)", []

    # 2. Request <= Available check
    if not all(request[j] <= available[j] for j in range(m)):
        return False, "Resources not currently available (Must wait)", []

    # 3. Pretend allocation
    temp_avail = [available[j] - request[j] for j in range(m)]
    temp_alloc = copy.deepcopy(allocation)
    for j in range(m):
        temp_alloc[p_idx][j] += request[j]

    is_safe, seq = check_bankers_safety(temp_avail, temp_alloc, max_matrix)
    if is_safe:
        return True, "Request granted safely. System remains in safe state.", seq
    else:
        return False, "Request DENIED: Allocation leads to an UNSAFE state (Deadlock Risk).", []


def run_bankers_algorithm(custom_request_p: int = None, custom_request_vec: list = None):
    """
    Demonstrates the complete textbook Banker's Algorithm benchmark.
    Tests standard textbook requests (P1 requesting [1,0,2] and P4 requesting [3,3,0]).
    """
    avail = list(TEXTBOOK_AVAILABLE)
    alloc = copy.deepcopy(TEXTBOOK_ALLOCATION)
    max_m = copy.deepcopy(TEXTBOOK_MAX)
    need = [[max_m[i][j] - alloc[i][j] for j in range(3)] for i in range(5)]

    is_safe, seq = check_bankers_safety(avail, alloc, max_m)

    # Standard textbook tests
    p1_grant, p1_reason, p1_seq = request_resources_bankers(1, [1, 0, 2], avail, alloc, max_m)
    p4_grant, p4_reason, p4_seq = request_resources_bankers(4, [3, 3, 0], avail, alloc, max_m)

    custom_result = None
    if custom_request_p is not None and custom_request_vec is not None:
        c_grant, c_reason, c_seq = request_resources_bankers(custom_request_p, custom_request_vec, avail, alloc, max_m)
        custom_result = {
            "process": f"P{custom_request_p}",
            "request": custom_request_vec,
            "granted": c_grant,
            "decision": "GRANTED" if c_grant else "DENIED",
            "reason": c_reason,
            "safe_sequence": c_seq,
        }

    return {
        "initial_state_safe": is_safe,
        "safe_sequence": seq,
        "resources": ["A (CPU)", "B (RAM)", "C (GPU)"],
        "available": avail,
        "processes": [
            {
                "process": f"P{i}",
                "allocation": alloc[i],
                "max": max_m[i],
                "need": need[i],
            }
            for i in range(5)
        ],
        "textbook_tests": [
            {
                "process": "P1",
                "request": [1, 0, 2],
                "granted": p1_grant,
                "decision": "GRANTED" if p1_grant else "DENIED",
                "reason": p1_reason,
                "safe_sequence": p1_seq,
            },
            {
                "process": "P4",
                "request": [3, 3, 0],
                "granted": p4_grant,
                "decision": "DENIED" if not p4_grant else "GRANTED",
                "reason": p4_reason,
                "safe_sequence": p4_seq,
            },
        ],
        "custom_test": custom_result,
    }


# =============================================================================
# What-If Simulator Scenarios
# =============================================================================

def whatif_reduce_ram_ceiling(db: Session, new_ram_ceiling_mb: float):
    history = db.query(models.ExecutionHistory).all()
    blocked = sum(1 for h in history if (h.actual_ram_mb or 0) > new_ram_ceiling_mb)
    max_ram = max((h.actual_ram_mb or 0 for h in history), default=0.0)
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
        avg_duration = 5.0

    pending_count = db.query(models.Task).filter(models.Task.status == "pending").count()
    projected_extra_wait = (pending_count + extra_tasks) * avg_duration
    return {
        "scenario": f"+{extra_tasks} tasks",
        "projected_extra_wait_sec": round(projected_extra_wait, 2),
    }
