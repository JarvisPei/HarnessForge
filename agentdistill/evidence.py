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
    output_format: str = typer.Option("json", "--format", "-f", help="Output format: json or markdown."),
) -> None:
    report = build_evidence_report(run_dirs)
    if output_format == "json":
        text = json.dumps(report, indent=2, ensure_ascii=False)
    elif output_format == "markdown":
        text = render_evidence_markdown(report)
    else:
        raise typer.BadParameter("format must be json or markdown")
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
    applied_types = patches.get("applied_type_counts", {})
    return {
        "run_dir": row.get("run_dir"),
        "run_name": _run_name(row.get("run_dir")),
        "accepted": _int(patches.get("accepted")),
        "accepted_runtime_artifact": _int(patches.get("accepted_runtime_artifact")),
        "accepted_test_only": _int(patches.get("accepted_test_only")),
        "accepted_but_no_runtime_artifact": _int(patches.get("accepted_but_no_runtime_artifact")),
        "artifact_types": _artifact_types(applied_types if isinstance(applied_types, dict) else {}),
        "dev_improved": _int(dev.get("improved")),
        "dev_regressed": _int(dev.get("regressed")),
        "blind_improved": _int(blind.get("improved")),
        "blind_regressed": _int(blind.get("regressed")),
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
    if _int(blind_runtime.get("improved_with_runtime_effect")) > 0:
        return "end_to_end_runtime_transfer"
    if _int(blind.get("improved")) > 0:
        return "end_to_end_harness_transfer"
    if _int(dev_runtime.get("after_runtime_effect")) <= 0 and _int(blind_runtime.get("after_runtime_effect")) <= 0:
        return "harness_artifact_not_triggered"
    if _int(blind_runtime.get("after_runtime_effect")) > 0 and _int(blind.get("improved")) <= 0:
        return "runtime_effect_without_blind_transfer"
    if _int(dev_runtime.get("improved_with_runtime_effect")) > 0:
        return "dev_runtime_transfer"
    if _int(dev.get("improved")) > 0:
        return "dev_harness_transfer"
    return "no_transfer"


def _aggregate_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    end_to_end_statuses = {"end_to_end_runtime_transfer", "end_to_end_harness_transfer"}
    return {
        "num_runs": len(rows),
        "end_to_end_transfer_runs": sum(1 for row in rows if row["evidence_status"] in end_to_end_statuses),
        "end_to_end_runtime_transfer_runs": sum(
            1 for row in rows if row["evidence_status"] == "end_to_end_runtime_transfer"
        ),
        "end_to_end_harness_transfer_runs": sum(
            1 for row in rows if row["evidence_status"] == "end_to_end_harness_transfer"
        ),
        "runtime_artifact_runs": sum(1 for row in rows if row["accepted_runtime_artifact"] > 0),
        "test_only_runs": sum(1 for row in rows if row["accepted_test_only"] > 0),
        "runtime_effect_runs": sum(1 for row in rows if row["dev_runtime_effect"] > 0 or row["blind_runtime_effect"] > 0),
        "blind_runtime_effect_runs": sum(1 for row in rows if row["blind_runtime_effect"] > 0),
        "dev_improved": sum(row["dev_improved"] for row in rows),
        "dev_regressed": sum(row["dev_regressed"] for row in rows),
        "blind_improved": sum(row["blind_improved"] for row in rows),
        "blind_regressed": sum(row["blind_regressed"] for row in rows),
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


def _run_name(run_dir: Any) -> str:
    if not isinstance(run_dir, str) or not run_dir:
        return ""
    return Path(run_dir).name


def _artifact_types(type_counts: dict[str, Any]) -> list[str]:
    return [name for name, count in sorted(type_counts.items()) if isinstance(count, int) and count > 0]


def render_evidence_markdown(report: dict[str, Any]) -> str:
    aggregate = report.get("aggregate", {})
    rows = report.get("runs", [])
    lines = [
        "# Evidence Suite",
        "",
        "## Aggregate",
        "",
        f"- runs: {_int(aggregate.get('num_runs'))}",
        f"- end-to-end transfer runs: {_int(aggregate.get('end_to_end_transfer_runs'))}",
        f"- runtime transfer runs: {_int(aggregate.get('end_to_end_runtime_transfer_runs'))}",
        f"- harness transfer runs: {_int(aggregate.get('end_to_end_harness_transfer_runs'))}",
        f"- dev improved/regressed: {_int(aggregate.get('dev_improved'))}/{_int(aggregate.get('dev_regressed'))}",
        f"- blind improved/regressed: {_int(aggregate.get('blind_improved'))}/{_int(aggregate.get('blind_regressed'))}",
        f"- blind improved with runtime effect: {_int(aggregate.get('blind_improved_with_runtime_effect'))}",
        "",
        "## Runs",
        "",
        "| run | status | artifacts | accepted | dev +/- | blind +/- | blind runtime effects | blind runtime wins |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            artifacts = ", ".join(row.get("artifact_types", [])) if isinstance(row.get("artifact_types"), list) else ""
            lines.append(
                "| "
                f"{_md_cell(str(row.get('run_name') or row.get('run_dir') or ''))} | "
                f"{_md_cell(str(row.get('evidence_status') or ''))} | "
                f"{_md_cell(artifacts)} | "
                f"{_int(row.get('accepted'))} | "
                f"{_int(row.get('dev_improved'))}/{_int(row.get('dev_regressed'))} | "
                f"{_int(row.get('blind_improved'))}/{_int(row.get('blind_regressed'))} | "
                f"{_int(row.get('blind_runtime_effect'))} | "
                f"{_int(row.get('blind_improved_with_runtime_effect'))} |"
            )
    return "\n".join(lines) + "\n"


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    app()
