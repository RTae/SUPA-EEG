"""Aggregate successful DVC experiment metrics into summary reports."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
CSV_PATH = REPORT_DIR / "experiment_results_summary.csv"
MD_PATH = REPORT_DIR / "experiment_results_summary.md"


def run_cmd(*args: str) -> str:
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def try_show_file(rev: str, path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def parse_result_text(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if key in {"top1", "top5"}:
            values[key] = float(raw_value)
        elif key == "epoch":
            values[key] = int(float(raw_value))
        else:
            values[key] = raw_value
    return values


def find_result_path(rev: str) -> str | None:
    paths = run_cmd("git", "ls-tree", "-r", "--name-only", rev).splitlines()
    for path in paths:
        if path.endswith("result.txt"):
            return path
    return None


def load_metrics(rev: str) -> dict[str, Any]:
    json_text = try_show_file(rev, "reports/dvc_metrics.json")
    if json_text:
        data = json.loads(json_text)
        if isinstance(data, dict):
            return data

    result_path = find_result_path(rev)
    if not result_path:
        return {}

    result_text = try_show_file(rev, result_path)
    if not result_text:
        return {}

    metrics = parse_result_text(result_text)
    if result_path:
        metrics["result_path"] = result_path
    return metrics


def load_params(rev: str) -> dict[str, Any]:
    text = try_show_file(rev, "params.yaml")
    if not text:
        return {}
    data = yaml.safe_load(text) or {}
    return data if isinstance(data, dict) else {}


def collect_rows() -> list[dict[str, Any]]:
    refs = run_cmd("git", "for-each-ref", "--format=%(refname:short)", "refs/exps").splitlines()
    rows: list[dict[str, Any]] = []

    for ref in refs:
        if "/celery/failed" in ref:
            continue

        rev = run_cmd("git", "rev-parse", ref)
        metrics = load_metrics(rev)
        if not {"top1", "top5", "epoch"}.issubset(metrics):
            continue

        params = load_params(rev)
        model = params.get("model", {}) if isinstance(params.get("model"), dict) else {}

        rows.append(
            {
                "experiment": ref.split("/")[-1],
                "rev": rev[:7],
                "subject": params.get("subject"),
                "metric": params.get("metric"),
                "model": model.get("name"),
                "feature_type": model.get("feature_type"),
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "epoch": metrics["epoch"],
                "result_path": metrics.get("result_path", "reports/dvc_metrics.json"),
            }
        )

    rows.sort(key=lambda row: row["top1"], reverse=True)
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "rev",
        "subject",
        "metric",
        "model",
        "feature_type",
        "top1",
        "top5",
        "epoch",
        "result_path",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Experiment Results Summary",
        "",
        f"Successful experiments aggregated: {len(rows)}",
        "",
        "| Experiment | Rev | Subject | Metric | Model | Feature | Top1 | Top5 | Epoch |",
        "|---|---|---:|---|---|---|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['rev']} | {row['subject']} | {row['metric']} | "
            f"{row['model']} | {row['feature_type']} | {row['top1']:.4f} | {row['top5']:.4f} | {row['epoch']} |"
        )

    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    write_csv(rows)
    write_markdown(rows)

    print(f"Wrote {len(rows)} successful experiments to {CSV_PATH.relative_to(ROOT)}")
    print(f"Wrote readable table to {MD_PATH.relative_to(ROOT)}")
    if rows:
        best = rows[0]
        print(
            "Best top1: "
            f"{best['experiment']} (subject={best['subject']}, top1={best['top1']:.4f}, top5={best['top5']:.4f})"
        )


if __name__ == "__main__":
    main()
