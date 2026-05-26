from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from agentdistill.repair_efficiency import build_repair_efficiency_report


app = typer.Typer(add_completion=False)


@app.command()
def main(
    run_dirs: list[Path] = typer.Argument(..., help="Benchmark run directories to summarize as evidence."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Optional JSON output path."),
) -> None:
    report = build_evidence_report(run_dirs)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    typer.echo(text)


def build_evidence_report(run_dirs: list[Path]) -> dict[str, Any]:
    efficiency = build_repair_efficiency_report(run_dirs)
    runs = [_evidence_row(row) for row in efficiency["runs"]]
    return {
        "runs": runs,
        "aggregate": _aggregate_evidence(runs),
    }


def _evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    patches = row.get("patches", {})
    dev = row.get("dev_transfer", {})
    blind = row.get("blind_transfer", {})
    runtime = row.get("runtime_effect", {})
    dev_runtime = runtime.get("dev", {}) if isinstance(runtime, dict) else {}
    blind_runtime = runtime.get("blind", {}) if isinstance(runtime, dict) else {}
    return {
        "run_dir": row.get("run_dir"),
        "accepted": _int(patches.get("accepted")),
        "accepted_runtime_artifact": _int(patches.get("accepted_runtime_artifact")),
        "accepted_test_only": _int(patches.get("accepted_test_only")),
        "accepted_but_no_runtime_artifact": _int(patches.get("accepted_but_no_runtime_artifact")),
        "dev_improved": _int(dev.get("improved")),
        "blind_improved": _int(blind.get("improved")),
        "dev_runtime_effect": _int(dev_runtime.get("after_runtime_effect")),
        "blind_runtime_effect": _int(blind_runtime.get("after_runtime_effect")),
        "dev_improved_with_runtime_effect": _int(dev_runtime.get("improved_with_runtime_effect")),
        "blind_improved_with_runtime_effect": _int(blind_runtime.get("improved_with_runtime_effect")),
        "evidence_status": _evidence_status(patches, dev, blind, dev_runtime, blind_runtime),
    }


def _evidence_status(
    patches: dict[str, Any],
    dev: dict[str, Any],
    blind: dict[str, Any],
    dev_runtime: dict[str, Any],
    blind_runtime: dict[str, Any],
) -> str:
    if _int(patches.get("accepted_runtime_artifact")) <= 0:
        return "no_runtime_artifact"
    if _int(dev_runtime.get("after_runtime_effect")) <= 0 and _int(blind_runtime.get("after_runtime_effect")) <= 0:
        return "runtime_artifact_not_triggered"
    if _int(blind_runtime.get("after_runtime_effect")) > 0 and _int(blind.get("improved")) <= 0:
        return "runtime_effect_without_blind_transfer"
    if _int(blind_runtime.get("improved_with_runtime_effect")) > 0:
        return "end_to_end_transfer"
    if _int(dev.get("improved")) > 0:
        return "dev_only_transfer"
    return "no_transfer"


def _aggregate_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_runs": len(rows),
        "end_to_end_transfer_runs": sum(1 for row in rows if row["evidence_status"] == "end_to_end_transfer"),
        "runtime_artifact_runs": sum(1 for row in rows if row["accepted_runtime_artifact"] > 0),
        "test_only_runs": sum(1 for row in rows if row["accepted_test_only"] > 0),
        "runtime_effect_runs": sum(1 for row in rows if row["dev_runtime_effect"] > 0 or row["blind_runtime_effect"] > 0),
        "blind_runtime_effect_runs": sum(1 for row in rows if row["blind_runtime_effect"] > 0),
        "blind_improved_with_runtime_effect": sum(row["blind_improved_with_runtime_effect"] for row in rows),
        "status_counts": _status_counts(rows),
    }


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["evidence_status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def _int(value: Any) -> int:
    return value if isinstance(value, int) else 0


if __name__ == "__main__":
    app()
