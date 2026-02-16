#!/usr/bin/env python3
"""
Bank generation progress monitor.

Stdlib only — runs instantly without conda activation.

Usage:
    python3 scripts/check_bank.py /data/matiasz/dot-pe/comparison_65k/bank_nrsur
    python3 scripts/check_bank.py /data/matiasz/dot-pe/comparison_65k/bank_nrsur bank_xphm
    python3 scripts/check_bank.py /data/matiasz/dot-pe/comparison_65k/bank_nrsur --verbose
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

MIN_FILE_BYTES = 200


def check_bank(bank_dir, verbose=False):
    bank_dir = Path(bank_dir)
    waveform_dir = bank_dir / "waveforms"
    config_path = bank_dir / "bank_config.json"

    if not config_path.exists():
        print(f"  ERROR: No bank_config.json in {bank_dir}")
        return

    with open(config_path) as f:
        cfg = json.load(f)

    approximant = cfg.get("approximant", "unknown")
    bank_size = cfg["bank_size"]
    blocksize = cfg.get("blocksize", 4096)
    n_blocks = bank_size // blocksize

    complete = []
    missing = []
    partial = []
    mtimes = []  # completion times for rate estimation

    for i in range(n_blocks):
        amp = waveform_dir / f"amplitudes_block_{i}.npy"
        pha = waveform_dir / f"phase_block_{i}.npy"
        amp_ok = amp.exists() and amp.stat().st_size > MIN_FILE_BYTES
        pha_ok = pha.exists() and pha.stat().st_size > MIN_FILE_BYTES

        if amp_ok and pha_ok:
            complete.append(i)
            # Use the later mtime of the two files
            mt = max(amp.stat().st_mtime, pha.stat().st_mtime)
            mtimes.append(mt)
        elif amp_ok or pha_ok:
            partial.append(i)
        else:
            missing.append(i)

    n_complete = len(complete)
    n_missing = len(missing)
    n_partial = len(partial)
    pct = 100.0 * n_complete / n_blocks if n_blocks > 0 else 0

    print(f"=== Bank: {bank_dir.name} ({approximant}) ===")
    print(f"  Total blocks:    {n_blocks}")
    print(f"  Complete:        {n_complete}")
    print(f"  Missing:         {n_missing}")
    print(f"  Partial/corrupt: {n_partial}")
    print(f"  Progress:        {n_complete}/{n_blocks} ({pct:.0f}%)")

    if missing:
        if len(missing) <= 20:
            print(f"  Missing indices: {missing}")
        else:
            print(f"  Missing indices: {missing[:10]} ... ({n_missing} total)")

    if partial:
        print(f"  Partial indices: {partial}")

    # Rate estimation from completion mtimes
    if len(mtimes) >= 2:
        mtimes.sort()
        # Time between first and last completion
        span = mtimes[-1] - mtimes[0]
        if span > 0 and n_complete > 1:
            avg_time = span / (n_complete - 1)
            print(f"  Avg time/block:  {avg_time:.0f} s")
            if n_missing > 0:
                eta_serial = avg_time * n_missing
                print(f"  ETA (serial):    {_fmt_duration(eta_serial)}")

    # Detect active processes
    _detect_active_procs(bank_dir)

    # Detect SLURM jobs
    _detect_slurm_jobs(bank_dir)

    if verbose and complete:
        # Show file sizes for first complete block
        i = complete[0]
        amp = waveform_dir / f"amplitudes_block_{i}.npy"
        pha = waveform_dir / f"phase_block_{i}.npy"
        print(f"  Sample file sizes (block {i}):")
        print(f"    amplitudes: {amp.stat().st_size / 1e6:.1f} MB")
        print(f"    phase:      {pha.stat().st_size / 1e6:.1f} MB")

    print()


def _fmt_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f} s"
    elif seconds < 3600:
        return f"{seconds / 60:.0f} min"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def _detect_active_procs(bank_dir):
    """Check for running gen_bank_block.py processes targeting this bank."""
    try:
        result = subprocess.run(
            ["pgrep", "-af", "gen_bank_block.py"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            bank_str = str(bank_dir)
            matching = [
                line for line in result.stdout.strip().split("\n")
                if bank_str in line
            ]
            if matching:
                print(f"  Active procs:    {len(matching)}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def _detect_slurm_jobs(bank_dir):
    """Check for SLURM array jobs targeting this bank."""
    bank_name = bank_dir.name  # e.g. "bank_nrsur"
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-n", f"bank_{bank_name}",
             "--noheader", "-o", "%A %t %M"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            running = sum(1 for l in lines if " R " in l)
            pending = sum(1 for l in lines if " PD " in l)
            print(f"  SLURM jobs:      {running} running, {pending} pending")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # squeue not available (not on Typhon)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 check_bank.py BANK_DIR [BANK_DIR2 ...] [--verbose]")
        sys.exit(1)

    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    bank_dirs = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    # Support relative names under a common parent
    resolved = []
    for bd in bank_dirs:
        p = Path(bd)
        if p.is_dir():
            resolved.append(p)
        else:
            # Try as sibling of first argument's parent
            if resolved:
                sibling = resolved[0].parent / bd
                if sibling.is_dir():
                    resolved.append(sibling)
                else:
                    print(f"WARNING: Not a directory: {bd}")
            else:
                print(f"WARNING: Not a directory: {bd}")

    for bd in resolved:
        check_bank(bd, verbose=verbose)


if __name__ == "__main__":
    main()
