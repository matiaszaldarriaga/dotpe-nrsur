"""
Standalone correctness checker: compare a run's summary_results.json against baseline.

Usage:
    python scripts/optim_correctness.py <results_dir>
    python scripts/optim_correctness.py  # defaults to optim/benchmark_run/results

Exits with code 0 on PASS, 1 on FAIL.
"""
import sys
import json
from pathlib import Path

OPTIM_DIR = Path(__file__).resolve().parent.parent / "optim"
BASELINE_JSON = OPTIM_DIR / "baseline.json"

# Tolerance definitions
CHECKS = [
    # (key, tolerance, is_relative, is_critical, description)
    ("bestfit_lnlike_max", 1.0, False, True,
     "Best-fit log-likelihood (deterministic given same templates)"),
    ("lnl_marginalized_max", 2.0, False, False,
     "Marginalized log-likelihood (QMC variance)"),
    ("ln_evidence", 5.0, False, False,
     "Log evidence (noisy integral)"),
    ("n_effective", 1.0, True, False,
     "Effective sample count (sensitive to exact sample set)"),
]

INJECTION_CHECKS = [
    ("bestfit_lnlike", 0.5, False, True,
     "Injection best-fit log-likelihood (should be stable)"),
]


def load_summary(results_dir):
    """Load summary_results.json from a results directory."""
    path = Path(results_dir) / "summary_results.json"
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(2)
    return json.load(open(path))


def run_checks(summary, baseline_correctness):
    """Run all correctness checks. Returns (critical_pass, all_pass, details)."""
    details = []

    for key, tol, is_relative, is_critical, desc in CHECKS:
        bl_val = baseline_correctness[key]
        cur_val = summary[key]

        if is_relative:
            diff = abs(cur_val - bl_val) / abs(bl_val) if bl_val != 0 else abs(cur_val)
            passed = diff < tol
            diff_str = f"{diff:.1%}"
            tol_str = f"{tol:.0%}"
        else:
            diff = abs(cur_val - bl_val)
            passed = diff < tol
            diff_str = f"{diff:.4f}"
            tol_str = f"{tol:.1f}"

        details.append({
            "metric": key,
            "baseline": bl_val,
            "current": cur_val,
            "diff": diff_str,
            "tolerance": tol_str,
            "passed": passed,
            "critical": is_critical,
            "description": desc,
        })

    # Injection checks
    for key, tol, is_relative, is_critical, desc in INJECTION_CHECKS:
        bl_val = baseline_correctness[f"injection_{key}"]
        cur_val = summary["injection"][key]

        diff = abs(cur_val - bl_val)
        passed = diff < tol
        diff_str = f"{diff:.4f}"
        tol_str = f"{tol:.1f}"

        details.append({
            "metric": f"injection.{key}",
            "baseline": bl_val,
            "current": cur_val,
            "diff": diff_str,
            "tolerance": tol_str,
            "passed": passed,
            "critical": is_critical,
            "description": desc,
        })

    critical_pass = all(d["passed"] for d in details if d["critical"])
    all_pass = all(d["passed"] for d in details)

    return critical_pass, all_pass, details


def main():
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = OPTIM_DIR / "benchmark_run" / "results"

    if not BASELINE_JSON.exists():
        print(f"ERROR: No baseline at {BASELINE_JSON}. Run: python scripts/run_baseline.py")
        sys.exit(2)

    baseline = json.load(open(BASELINE_JSON))
    summary = load_summary(results_dir)

    critical_pass, all_pass, details = run_checks(summary, baseline["correctness"])

    # Print results
    print(f"\n{'=' * 80}")
    print(f"CORRECTNESS CHECK: {results_dir}")
    print(f"{'=' * 80}")
    print(f"\n{'Metric':<30s} {'Baseline':>12s} {'Current':>12s} {'Diff':>10s} {'Tol':>8s} {'Result':>8s}")
    print("-" * 80)

    for d in details:
        status = "PASS" if d["passed"] else "*** FAIL ***"
        crit = " [C]" if d["critical"] else ""
        print(f"  {d['metric']:<28s} {d['baseline']:>12.3f} {d['current']:>12.3f} "
              f"{d['diff']:>10s} {d['tolerance']:>8s} {status}{crit}")

    print()
    print(f"  [C] = critical check (must pass)")
    print()

    if critical_pass and all_pass:
        print("  VERDICT: PASS (all checks)")
    elif critical_pass:
        print("  VERDICT: PASS (critical checks; some non-critical outside tolerance)")
    else:
        print("  VERDICT: *** FAIL *** (critical check failed!)")
        print("  This indicates a real bug — revert the change.")

    sys.exit(0 if critical_pass else 1)


if __name__ == "__main__":
    main()
