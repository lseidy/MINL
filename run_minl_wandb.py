from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

try:
    import wandb
except Exception:
    wandb = None


EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)\s+avg_loss=([0-9.eE+-]+)\s+loss_l2=([0-9.eE+-]+)\s+l1_reg=([0-9.eE+-]+)\s+lr=([0-9.eE+-]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=str, default="MINL", help="wandb project name")
    p.add_argument("--name", type=str, default=None, help="wandb run name")
    p.add_argument("--entity", type=str, default=None, help="wandb entity")
    p.add_argument("--api-key", type=str, default=None, help="wandb api key (optional)")
    p.add_argument("--minl-py", type=str, default="minl_new.py", help="path to minl_new.py (relative to this script)")
    # pass-through separator: everything after '--' goes to minl_new.py
    p.add_argument("--", dest="_", nargs=argparse.REMAINDER)
    return p.parse_args()


def run_minl_and_log(minl_path: Path, pass_args: List[str], project: str, name: Optional[str], entity: Optional[str], api_key: Optional[str]):
    # init wandb
    if wandb is None:
        print("wandb is not installed. Install it or remove run_minl_wandb usage.")
    else:
        if api_key:
            try:
                wandb.login(key=api_key, force=True)
            except Exception:
                wandb.login()
        else:
            try:
                wandb.login()
            except Exception:
                print("wandb login failed or skipped; proceeding without login")

    cmd = [sys.executable, str(minl_path)] + pass_args
    print("Running:", " ".join(shlex.quote(c) for c in cmd))

    # start subprocess
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    run = None
    if wandb is not None:
        run = wandb.init(project=project, name=name, entity=entity, reinit=True)

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            print(line)
            m = EPOCH_RE.search(line)
            if m:
                epoch = int(m.group(1))
                total = int(m.group(2))
                avg_loss = float(m.group(3))
                loss_l2 = float(m.group(4))
                l1_reg = float(m.group(5))
                lr = float(m.group(6))
                if run is not None:
                    wandb.log({"epoch": epoch, "avg_loss": avg_loss, "loss_l2": loss_l2, "l1_reg": l1_reg, "lr": lr}, step=epoch)
        rc = proc.wait()
    finally:
        if run is not None:
            wandb.finish()

    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def main():
    args = parse_args()
    extra = args._ or []
    script_path = Path(args.minl_py)
    if not script_path.is_absolute():
        script_path = Path(__file__).parent.joinpath(script_path).resolve()
    if not script_path.exists():
        print(f"minl script not found: {script_path}")
        sys.exit(1)

    run_minl_and_log(script_path, extra, project=args.project, name=args.name, entity=args.entity, api_key=args.api_key)


if __name__ == "__main__":
    main()
