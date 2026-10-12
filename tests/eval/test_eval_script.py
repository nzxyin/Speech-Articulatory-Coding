"""``scripts/slurm/eval.sh``: item list, DONE markers, requeue on exit code 75, with stub ``python``/``srun``/``scontrol``.

The script is run by ``bash`` outside Slurm: ``srun``, ``scontrol`` and ``sleep`` are stubs on ``PATH`` and the venv python
is a stub that records its arguments and writes the DONE marker the way the real CLI does (none for aggregate/samples).
"""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "slurm" / "eval.sh"

PY_STUB = """#!/bin/bash
# stub of `python -m sparc.cli.eval_vocoder stage=.. system=.. condition=..`
echo "$*" >> "$STUB_LOG"
for a in "$@"; do case "$a" in stage=*) stage=${a#stage=};; system=*) system=${a#system=};; condition=*) cond=${a#condition=};; esac; done
rc=0
[ -n "${STUB_RC_FOR:-}" ] && [ "$stage:$system:$cond" = "$STUB_RC_FOR" ] && rc=${STUB_RC:-0}
if [ "$rc" -eq 0 ]; then
    case "$stage" in aggregate|samples) ;; *) mkdir -p "$SV_ROOT/eval${LIMIT_DIR:-}/done/test.clean"; date > "$SV_ROOT/eval${LIMIT_DIR:-}/done/test.clean/${stage}__${system}__${cond}";; esac
fi
exit $rc
"""


@pytest.fixture
def sandbox(tmp_path):
    sv = tmp_path / "sv"
    (tmp_path / "bin").mkdir()
    (tmp_path / "repo" / ".venv" / "bin").mkdir(parents=True)
    env_file = tmp_path / "env.sh"
    env_file.write_text(f"export SV_ROOT={sv}\nexport SPARC_VOC_RUNS={sv}/runs\n")
    python = tmp_path / "repo" / ".venv" / "bin" / "python"
    python.write_text(PY_STUB)
    stubs = {
        "srun": 'while [ "${1#--}" != "$1" ]; do shift; done\nexec "$@"\n',  # drop srun options, run the command
        "scontrol": 'echo "scontrol $*" >> "$STUB_LOG"\n',
        "sleep": "exit 0\n",
    }
    for name, body in stubs.items():
        (tmp_path / "bin" / name).write_text("#!/bin/bash\n" + body)
    for path in [python, *(tmp_path / "bin").iterdir()]:
        path.chmod(0o755)
    log = tmp_path / "stub.log"
    env = {
        **os.environ,
        "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        "SPARC_ENV_FILE": str(env_file),
        "SV_EVAL_REPO": str(tmp_path / "repo"),
        "SV_EVAL_CPU": "1",
        "SLURM_JOB_ID": "123",
        "SLURMD_NODENAME": "node",
        "STUB_LOG": str(log),
    }
    env.pop("SV_EVAL_LIMIT", None)

    def run(*args: str, **extra: str) -> tuple[subprocess.CompletedProcess, list[str]]:
        done = subprocess.run(["bash", str(SCRIPT), *args], env={**env, **extra}, capture_output=True, text=True, timeout=60)
        calls = log.read_text().splitlines() if log.is_file() else []
        log.unlink(missing_ok=True)
        return done, calls

    run.done_dir = sv / "eval" / "done" / "test.clean"
    return run


def stages_of(calls: list[str]) -> list[str]:
    return [c.split("stage=")[1].split()[0] + ":" + c.split("system=")[1].split()[0] for c in calls if "stage=" in c]


def test_items_run_in_order_and_finished_ones_are_skipped(sandbox):
    done, calls = sandbox("gt:gt:T1", "signal:all:all", "eval.require_final=false")
    assert done.returncode == 0, done.stdout + done.stderr
    assert stages_of(calls) == ["gt:gt", "signal:all"]
    assert all("eval.require_final=false" in c and "eval.split=test.clean" in c and "eval.device=cpu" in c for c in calls)
    assert (sandbox.done_dir / "gt__gt__T1").is_file()
    done, calls = sandbox("gt:gt:T1", "signal:all:all", "utmos:all:all")
    assert done.returncode == 0
    assert stages_of(calls) == ["utmos:all"]  # the first two are finished: their markers exist


def test_aggregate_and_samples_run_every_time_even_with_a_stale_marker(sandbox):
    sandbox.done_dir.mkdir(parents=True)
    (sandbox.done_dir / "aggregate__all__all").write_text("2026-10-05\n")  # left by an older version of the CLI
    (sandbox.done_dir / "gt__gt__T1").write_text("2026-10-05\n")
    done, calls = sandbox("gt:gt:T1", "aggregate:all:all")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "all items already done" not in done.stdout
    assert stages_of(calls) == ["aggregate:all"]
    done, calls = sandbox("aggregate:all:all", "samples:all:all")
    assert stages_of(calls) == ["aggregate:all", "samples:all"]  # no marker is written, so both run again


def test_exit_code_75_requeues_the_job_and_stops_the_list(sandbox):
    done, calls = sandbox("gt:gt:T1", "signal:all:all", "utmos:all:all", STUB_RC_FOR="signal:all:all", STUB_RC="75")
    assert done.returncode == 75
    assert "scontrol requeue 123" in calls
    assert stages_of(calls) == ["gt:gt", "signal:all"]  # utmos never started
    assert not (sandbox.done_dir / "signal__all__all").exists()


def test_failure_stops_the_list_without_requeue(sandbox):
    done, calls = sandbox("gt:gt:T1", "signal:all:all", "utmos:all:all", STUB_RC_FOR="signal:all:all", STUB_RC="3")
    assert done.returncode == 3
    assert not any(c.startswith("scontrol") for c in calls)
    assert stages_of(calls) == ["gt:gt", "signal:all"]


def test_smoke_limit_uses_its_own_done_directory(sandbox):
    done, calls = sandbox("gt:gt:T1", SV_EVAL_LIMIT="4", LIMIT_DIR="/smoke_4")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "eval.limit=4" in calls[0]
    assert (sandbox.done_dir.parent.parent.parent / "eval" / "smoke_4" / "done" / "test.clean" / "gt__gt__T1").is_file()
    assert not (sandbox.done_dir / "gt__gt__T1").exists()


def test_bad_arguments_are_rejected(sandbox):
    assert sandbox("gt")[0].returncode == 2
    assert sandbox()[0].returncode == 2
