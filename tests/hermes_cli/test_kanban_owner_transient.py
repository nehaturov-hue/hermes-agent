"""Owner-transient worker exits are neutral to the failure budget, and bounded.

A dispatcher-spawned Kanban worker whose gateway owner stays silent (a loop stall, a restart that
outlasts the owner-silence budget) or whose admission reads ``unknown`` is a harness failure, not a
work failure. The worker client exits ``KANBAN_OWNER_TRANSIENT_EXIT_CODE`` (76), and the dispatcher
books that run ``owner_transient``: the card returns to its source phase, ``consecutive_failures``
does not move, and the next respawn waits ``kanban.owner_transient_cooldown_seconds``. The class
is bounded: after ``kanban.owner_transient_limit`` consecutive free exits the next one is an
ordinary crash. A genuine failure keeps today's accounting.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import kanban_watchers_notifier as notifier
from gateway import session_kanban
from hermes_cli import gateway_client, kanban_worker_client
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

OWNER_TRANSIENT = (
    76  # the contract number: EX_PROTOCOL in sysexits.h, free in this codebase
)
GUE = gateway_client.GatewayUnavailableError
_PIDS = itertools.count(70000)


# --- fixtures and helpers -------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (mirrors tests/hermes_cli/test_kanban_db.py)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture(autouse=True)
def _dead_workers(monkeypatch):
    """Every pid this file books is dead, the launch grace is off, and the exit registry is private."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kbd, "_recent_worker_exits", {})


def _die_with(conn, tid, code):
    """One worker death: the task is claimed, its worker exits ``code``, the reclaim sweep books it."""
    pid = next(_PIDS)
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tid))
    conn.commit()
    kbd._record_worker_exit(pid, code << 8)
    return kbd.detect_crashed_workers(conn)


def _finish_run(conn, tid, outcome):
    """A run that ended on its own (completed, or any other non-death outcome)."""
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:done{next(_PIDS)}")
    run_id = kb.get_task(conn, tid).current_run_id
    conn.execute(
        "UPDATE task_runs SET outcome=?, status=?, ended_at=? WHERE id=?",
        (outcome, outcome, int(time.time()), run_id),
    )
    conn.execute(
        "UPDATE tasks SET status='ready', current_run_id=NULL, claim_lock=NULL, "
        "claim_expires=NULL, worker_pid=NULL WHERE id=?",
        (tid,),
    )
    conn.commit()


def _outcomes(conn, tid):
    return [
        r["outcome"]
        for r in conn.execute(
            "SELECT outcome FROM task_runs WHERE task_id=? AND ended_at IS NOT NULL ORDER BY id",
            (tid,),
        )
    ]


def _kinds(conn, tid):
    return [
        r["kind"]
        for r in conn.execute("SELECT kind FROM task_events WHERE task_id=?", (tid,))
    ]


def _last_event_payload(conn, tid, kind):
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind=? ORDER BY id DESC LIMIT 1",
        (tid, kind),
    ).fetchone()
    return json.loads(row["payload"])


def _config(monkeypatch, kanban_cfg):
    """Replace the config the dispatcher reads for ``kanban.*`` keys."""
    from hermes_cli import config

    monkeypatch.setattr(config, "load_config_readonly", lambda: {"kanban": kanban_cfg})


# --- the worker client's exit mapping, through main() ---------------------------------------------


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(kanban_worker_client, "time", clock)

    async def _sleep(seconds):
        clock.now += seconds

    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _sleep)
    return clock


@pytest.fixture
def owner_env(monkeypatch, tmp_path):
    """The environment ``kanban_worker_client._submit`` reads, and a worker result that must not be
    reached on the owner-transient paths."""
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    for key, value in {
        "HERMES_KANBAN_BOARD": "jobseek-test",
        "HERMES_KANBAN_TASK": "t_test",
        "HERMES_KANBAN_RUN_ID": "7",
        "HERMES_KANBAN_CLAIM_LOCK": "host:1",
        "HERMES_KANBAN_DB": str(tmp_path / "kanban.db"),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        session_kanban,
        "worker_result",
        lambda *_: {"exit_code": 1, "last_output": "card closed"},
    )
    return tmp_path


class _Owner:
    """Scripted gateway owner: ``receipts`` answer successive ``prompt.receipt`` calls (an exception
    step is raised), ``kanban.run`` always admits the attempt, and ``dial_error`` fails every dial."""

    def __init__(self, receipts=(), dial_error=None):
        self.receipts = list(receipts)
        self.dial_error = dial_error

    def connect(self):
        owner = self

        @asynccontextmanager
        async def _connect():
            if owner.dial_error is not None:
                raise owner.dial_error

            class _Client:
                def __init__(self):
                    self.events = asyncio.Queue()

                async def rpc(self, method, **params):
                    if method == "kanban.run":
                        return {
                            "session_id": "s1",
                            "receipt": {"status": "pending", "admission_id": "a1"},
                        }
                    step = owner.receipts.pop(0) if owner.receipts else "terminal"
                    if isinstance(step, Exception):
                        raise step
                    return {"status": step}

            yield _Client()

        return _connect


def _owner_silent():
    """A receipt read that never settles: the owner keeps refusing to answer."""
    return gateway_client.GatewayClientError(
        "Gateway connection failed; no local fallback"
    )


def test_unknown_receipt_exits_76_and_prints_the_trailer(
    monkeypatch, owner_env, capsys
):
    monkeypatch.setattr(
        gateway_client, "connect_gateway", _Owner(receipts=["unknown"]).connect()
    )

    assert kanban_worker_client.main() == OWNER_TRANSIENT
    assert "[kanban-worker-exit] rc=76" in capsys.readouterr().err


def test_owner_silent_for_the_whole_budget_exits_76(
    monkeypatch, owner_env, clock, capsys
):
    owner = _Owner(receipts=[_owner_silent()] * 400)
    monkeypatch.setattr(gateway_client, "connect_gateway", owner.connect())

    assert kanban_worker_client.main() == OWNER_TRANSIENT
    assert clock.now >= kanban_worker_client.OWNER_WAIT_S
    assert "[kanban-worker-exit] rc=76" in capsys.readouterr().err


def test_terminal_result_keeps_the_workers_own_exit_code(
    monkeypatch, owner_env, capsys
):
    monkeypatch.setattr(
        gateway_client, "connect_gateway", _Owner(receipts=["terminal"]).connect()
    )

    assert kanban_worker_client.main() == 1
    assert "[kanban-worker-exit] rc=1" in capsys.readouterr().err


@pytest.mark.parametrize(
    "case",
    ["missing-env", "local-owner-only", "refused-authorization", "owner-refusal"],
)
def test_unmapped_client_failures_still_exit_1(monkeypatch, owner_env, case):
    owner = _Owner()
    if case == "missing-env":
        monkeypatch.delenv("HERMES_KANBAN_TASK")
    elif case == "local-owner-only":
        monkeypatch.setenv("HERMES_TUI_GATEWAY_URL", "ws://127.0.0.1:9/")
    elif case == "refused-authorization":
        owner = _Owner(
            dial_error=GUE(
                "Gateway inaccessible: authorization", "inaccessible", "authorization"
            )
        )
    else:
        owner = _Owner(receipts=[gateway_client.GatewayRPCError("claim_mismatch")])
    monkeypatch.setattr(gateway_client, "connect_gateway", owner.connect())

    assert kanban_worker_client.main() == 1


# --- the exit code and its decoding ---------------------------------------------------------------


def test_exit_76_is_the_owner_transient_contract(kanban_home):
    assert kb.KANBAN_OWNER_TRANSIENT_EXIT_CODE == OWNER_TRANSIENT
    assert kb.KANBAN_OWNER_TRANSIENT_EXIT_CODE not in {
        kb.KANBAN_RATE_LIMIT_EXIT_CODE,
        kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE,
    }


def test_reaped_exit_76_decodes_as_owner_transient(kanban_home):
    kbd._record_worker_exit(70500, OWNER_TRANSIENT << 8)

    assert kbd._classify_worker_exit(70500) == ("owner_transient", OWNER_TRANSIENT)


# --- accounting: the dispatcher books the exit, the card's budget is untouched --------------------


def test_two_owner_transient_exits_requeue_without_blocking(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stall", assignee="a")
        for _ in range(2):
            assert tid not in _die_with(conn, tid, OWNER_TRANSIENT)
            task = kb.get_task(conn, tid)
            assert task.status == "ready"
            assert task.consecutive_failures == 0
        assert "gave_up" not in _kinds(conn, tid) and "blocked" not in _kinds(conn, tid)
        assert _outcomes(conn, tid) == ["owner_transient", "owner_transient"]


def test_second_respawn_waits_for_the_cooldown_then_is_allowed(
    kanban_home, monkeypatch
):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stall-guard", assignee="a")
        _die_with(conn, tid, OWNER_TRANSIENT)
        ended = conn.execute(
            "SELECT ended_at FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()[0]

        monkeypatch.setattr(kb.time, "time", lambda: ended + 30)
        assert kbd.check_respawn_guard(conn, tid) == "owner_transient_cooldown"
        monkeypatch.setattr(kb.time, "time", lambda: ended + 61)
        assert kbd.check_respawn_guard(conn, tid) is None


def test_five_owner_transient_exits_in_one_tick_trip_no_breaker(kanban_home):
    with kbc.connect() as conn:
        tids = [
            kb.create_task(conn, title=f"stall-{i}", assignee="a") for i in range(5)
        ]
        for tid in tids:
            host = kb._claimer_id().split(":", 1)[0]
            pid = next(_PIDS)
            kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
            conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tid))
            kbd._record_worker_exit(pid, OWNER_TRANSIENT << 8)
        conn.commit()

        assert kbd.detect_crashed_workers(conn) == []
        assert kbd.detect_crashed_workers._last_auto_blocked == []
        for tid in tids:
            task = kb.get_task(conn, tid)
            assert task.status == "ready" and task.consecutive_failures == 0
            assert "sticky" not in (task.last_failure_error or "")


def test_seventh_consecutive_owner_transient_exit_counts_as_a_failure(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="poison", assignee="a")
        for _ in range(6):
            assert tid not in _die_with(conn, tid, OWNER_TRANSIENT)
            assert kb.get_task(conn, tid).consecutive_failures == 0

        assert tid in _die_with(conn, tid, OWNER_TRANSIENT)
        task = kb.get_task(conn, tid)
        assert task.consecutive_failures == 1 and task.status == "ready"
        payload = _last_event_payload(conn, tid, "crashed")
        assert payload["owner_transient_exhausted"] is True
        assert payload["streak"] == 7


def test_two_genuine_crashes_still_give_up(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="genuine", assignee="a")
        _die_with(conn, tid, 1)
        assert kb.get_task(conn, tid).status == "ready"
        _die_with(conn, tid, 1)

        assert kb.get_task(conn, tid).status == "blocked"
        assert "gave_up" in _kinds(conn, tid)


def test_genuine_owner_transient_genuine_gives_up_at_the_second_genuine(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="mixed", assignee="a")
        _die_with(conn, tid, 1)
        _die_with(conn, tid, OWNER_TRANSIENT)
        task = kb.get_task(conn, tid)
        assert task.status == "ready" and task.consecutive_failures == 1
        _die_with(conn, tid, 1)

        assert kb.get_task(conn, tid).status == "blocked"
        assert "gave_up" in _kinds(conn, tid)


def test_a_completed_run_resets_the_owner_transient_streak(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="reset", assignee="a")
        for _ in range(6):
            _die_with(conn, tid, OWNER_TRANSIENT)
        _finish_run(conn, tid, "completed")

        assert tid not in _die_with(conn, tid, OWNER_TRANSIENT)
        task = kb.get_task(conn, tid)
        assert task.status == "ready" and task.consecutive_failures == 0
        assert _outcomes(conn, tid)[-1] == "owner_transient"


def test_limit_zero_switches_the_class_off(kanban_home, monkeypatch):
    _config(monkeypatch, {"owner_transient_limit": 0})
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="off", assignee="a")
        _die_with(conn, tid, OWNER_TRANSIENT)
        assert kb.get_task(conn, tid).consecutive_failures == 1
        _die_with(conn, tid, OWNER_TRANSIENT)

        assert kb.get_task(conn, tid).status == "blocked"
        assert _outcomes(conn, tid) == ["crashed", "crashed"]


def test_cooldown_zero_respawns_at_once(kanban_home, monkeypatch):
    _config(monkeypatch, {"owner_transient_cooldown_seconds": 0})
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="no-cooldown", assignee="a")
        _die_with(conn, tid, OWNER_TRANSIENT)

        assert kbd.check_respawn_guard(conn, tid) is None


def test_owner_transient_run_is_neutral_to_the_protocol_violation_budget(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="violation-then-stall", assignee="a")
        _finish_run(conn, tid, "crashed")
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = (SELECT MAX(id) FROM task_runs WHERE task_id = ?)",
            (json.dumps({"protocol_violation": True}), tid),
        )
        _finish_run(conn, tid, "owner_transient")
        # A gateway stall says nothing about the card's answer: the violation streak is still one.
        assert kbd._protocol_violation_streak(conn, tid) == 1


# --- the notifier: an owner-transient event wakes nobody and renders as silence -------------------


def test_owner_transient_is_not_a_wake_or_ping_kind():
    assert "owner_transient" not in notifier.TERMINAL_KINDS
    assert "owner_transient" not in notifier._WAKE_KINDS
    assert "owner_transient" not in notifier._EVENT_FORMATTERS
    # An unknown kind is silent: the formatter returns None before it reads any state.
    formatter = notifier._KanbanNotification.__new__(notifier._KanbanNotification)
    assert (
        formatter.format_event(SimpleNamespace(kind="owner_transient", payload={}))
        is None
    )


# --- back-test: the operator's seven give-ups, replayed through the real reclaim sweep -------------
#
# Each sequence is copied from the card's own run table (kanban_show, board jobseek-review2): every
# run in order. ``stall`` is a worker that ended on the owner-stall signature (``execution unknown``
# or ``Gateway connection failed``), ``genuine`` a worker that failed on its own (an ImportError), and
# the other kinds are runs that ended without a worker death. The old client exits 1 on a stall, the
# new one 76; both replays go through ``detect_crashed_workers``. A block followed by its unblock
# resets the counter (kanban_db.py ``unblock_task``), a handoff does not, a completed run does.

BACKTEST = {
    "t_1e92504d-a": [
        ("stall", 3278),
        ("stall", 3279),
    ],  # 2026-11-03 17:54, two in a row
    "t_1e92504d-b": [
        ("stall", 3362),
        ("stall", 3364),
    ],  # 2026-11-04 04:46 and 05:13, two in a row
    "t_9c83e9f0": [
        ("stall", 3349),
        ("stall", 3350),
    ],  # 'Gateway connection failed', then 'unknown'
    "t_79aaaf32": [
        ("stall", 3337),
        ("stall", 3352),
    ],  # 03:42:25 and 04:46:03, two in a row
    "t_fec14639": [("stall", 3347), ("changes_requested", 3355), ("stall", 3359)],
    "t_d0624e2e": [
        ("stall", 3273),
        ("blocked", 3285),
        ("review_requested", 3332),
        ("changes_requested", 3336),
        ("blocked", 3338),
        ("stall", 3360),
    ],
}
# Old client (exit 1): the run table's stall pairs give up. t_d0624e2e does not: its two stalls are
# separated by unblocks, and each unblock resets the counter (see the receipt's back-test finding).
OLD_GIVES_UP = {
    "t_1e92504d-a": True,
    "t_1e92504d-b": True,
    "t_9c83e9f0": True,
    "t_79aaaf32": True,
    "t_fec14639": True,
    "t_d0624e2e": False,
}
GENUINE_PAIR = [("genuine", 3138), ("genuine", 3139)]  # t_8c5077ef, 2026-11-03 00:53


def _replay(conn, tid, steps, stall_code):
    for kind, _run in steps:
        if kind in ("stall", "genuine"):
            _die_with(conn, tid, stall_code if kind == "stall" else 1)
            if kb.get_task(conn, tid).status == "blocked":
                return
        elif kind == "blocked":
            _finish_run(conn, tid, "blocked")
            conn.execute(
                "UPDATE tasks SET consecutive_failures = 0 WHERE id = ?", (tid,)
            )
            conn.commit()
        else:
            _finish_run(conn, tid, kind)


@pytest.mark.parametrize("name", sorted(BACKTEST))
def test_backtest_stall_runs_are_owner_transient_under_the_new_exit(kanban_home, name):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=name, assignee="a")
        _replay(conn, tid, BACKTEST[name], stall_code=OWNER_TRANSIENT)

        task = kb.get_task(conn, tid)
        assert task.status == "ready" and task.consecutive_failures == 0, name
        assert "gave_up" not in _kinds(conn, tid)
        stalls = sum(1 for kind, _ in BACKTEST[name] if kind == "stall")
        assert _outcomes(conn, tid).count("owner_transient") == stalls


@pytest.mark.parametrize("name", sorted(BACKTEST))
def test_backtest_the_same_runs_under_the_old_exit_reproduce_the_give_ups(
    kanban_home, name
):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title=name, assignee="a")
        _replay(conn, tid, BACKTEST[name], stall_code=1)

        assert (kb.get_task(conn, tid).status == "blocked") is OLD_GIVES_UP[name], name
        assert ("gave_up" in _kinds(conn, tid)) is OLD_GIVES_UP[name]


def test_backtest_genuine_pair_still_gives_up_under_the_new_exit(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t_8c5077ef", assignee="a")
        _replay(conn, tid, GENUINE_PAIR, stall_code=OWNER_TRANSIENT)

        assert kb.get_task(conn, tid).status == "blocked"
        assert "gave_up" in _kinds(conn, tid)
        assert _outcomes(conn, tid) == ["crashed", "crashed"]


# --- the operator's renderers accept an owner-transient run and event ----------------------------


def test_cli_runs_log_and_show_render_an_owner_transient_run(kanban_home, capsys):
    from argparse import Namespace
    from hermes_cli import kanban as kanban_cli

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="render", assignee="a")
        _die_with(conn, tid, OWNER_TRANSIENT)
    # The worker's log ends with the exit trailer the worker client writes on every exit path.
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "Kanban owner refused: Kanban execution unknown; retained by owner\n"
        "[kanban-worker-exit] rc=76\n",
        encoding="utf-8",
    )

    runs = Namespace(
        task_id=tid, json=False, board=None, dispatch_board=None, state=None
    )
    assert kanban_cli._cmd_runs(runs) == 0
    assert "owner_transient" in capsys.readouterr().out
    show = Namespace(
        task_id=tid, json=False, board=None, dispatch_board=None, state=None
    )
    assert kanban_cli._cmd_show(show) == 0
    assert "owner_transient" in capsys.readouterr().out
    assert kanban_cli._cmd_log(Namespace(task_id=tid, tail=None)) == 0
    assert "[kanban-worker-exit] rc=76" in capsys.readouterr().out


# --- the tick: the respawn waits out the cooldown, and a review-lane exit returns to review ------


def test_the_tick_withholds_the_respawn_until_the_cooldown_elapses(
    kanban_home, monkeypatch
):
    # The assignee here is no installed profile: the tick would file the card as non-spawnable
    # before the respawn guard runs. The profile check is not what this test is about.
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 4242

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="tick-stall", assignee="a")
        for _ in range(2):
            _die_with(conn, tid, OWNER_TRANSIENT)
        assert kb.get_task(conn, tid).consecutive_failures == 0
        ended = conn.execute(
            "SELECT ended_at FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()[0]

        monkeypatch.setattr(kb.time, "time", lambda: ended + 30)
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert (tid, "owner_transient_cooldown") in res.respawn_guarded
        assert spawns == []

        monkeypatch.setattr(kb.time, "time", lambda: ended + 61)
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert tid in [row[0] for row in res.spawned]
        assert spawns == [tid]
        assert kb.get_task(conn, tid).consecutive_failures == 0


def test_an_owner_transient_exit_from_the_review_lane_returns_to_review(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review-stall", assignee="a")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='review' WHERE id=?", (tid,))
        host = kb._claimer_id().split(":", 1)[0]
        assert (
            kb.claim_review_task(conn, tid, claimer=f"{host}:rv{next(_PIDS)}")
            is not None
        )
        pid = next(_PIDS)
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tid))
        conn.commit()
        kbd._record_worker_exit(pid, OWNER_TRANSIENT << 8)

        assert tid not in kbd.detect_crashed_workers(conn)
        task = kb.get_task(conn, tid)
        assert task.status == "review" and task.consecutive_failures == 0
        assert _outcomes(conn, tid) == ["owner_transient"]


def test_the_tick_summary_names_an_owner_transient_release():
    assert (
        kbd.describe_suppression([kbd.DispatchResult(owner_transient=["t_x"])])
        == "owner_transient=1"
    )


def test_owner_transient_keeps_the_open_pr_guard_after_the_cooldown(
    kanban_home, monkeypatch
):
    # A crash after a PR was opened is held by the open-PR guard. An owner-transient exit keeps that
    # guard once its cooldown has passed: a second worker would open a duplicate PR.
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stall-after-pr", assignee="a")
        kb.add_comment(
            conn,
            tid,
            "a",
            "PR opened: https://github.com/NousResearch/hermes-agent/pull/1",
        )
        _die_with(conn, tid, OWNER_TRANSIENT)
        ended = conn.execute(
            "SELECT ended_at FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()[0]
        monkeypatch.setattr(kb.time, "time", lambda: ended + 61)
        assert kbd.check_respawn_guard(conn, tid) == "active_pr"


def test_owner_transient_skips_the_text_blocker_after_the_cooldown(
    kanban_home, monkeypatch
):
    # Text is never the diagnosis for a stall: a quota error left in last_failure_error by an earlier
    # run must not park the card as blocker_auth once the owner-transient cooldown has passed.
    stale = "quota exceeded for this account"
    assert kbd._RESPAWN_BLOCKER_RE.search(stale)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stall-stale-quota", assignee="a")
        conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (stale, tid))
        conn.commit()
        _die_with(conn, tid, OWNER_TRANSIENT)
        ended = conn.execute(
            "SELECT ended_at FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()[0]
        monkeypatch.setattr(kb.time, "time", lambda: ended + 61)
        assert kbd.check_respawn_guard(conn, tid) is None
