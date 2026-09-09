"""Create the mandatory, filesystem-only archive for one reproducible run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_DIRECTORIES = ("logs", "outputs")


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def initialize_run(project_root: Path, run_id: str, stage: str, purpose: str) -> Path:
    """Create a run archive without overwriting an existing manifest."""
    run_dir = project_root / "outputs" / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Run archive already exists: {run_dir}")
    for directory in REQUIRED_DIRECTORIES:
        (run_dir / directory).mkdir(parents=True, exist_ok=False)

    manifest = {
        "run_id": run_id,
        "stage": stage,
        "purpose": purpose,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operator": os.environ.get("USER"),
        "hostname": platform.node(),
        "git_commit": _git_commit(project_root),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "status": "initialized",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text("", encoding="utf-8")
    (run_dir / "report.md").write_text(
        f"# {run_id}\n\n阶段：{stage}\n\n目的：{purpose}\n\n## 结果\n\n待运行。\n",
        encoding="utf-8",
    )
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize a mandatory MOPD run archive")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--purpose", required=True)
    args = parser.parse_args()
    print(initialize_run(args.project_root.resolve(), args.run_id, args.stage, args.purpose))
