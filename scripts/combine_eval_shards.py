#!/usr/bin/env python3
"""Combine sharded FORGE evaluation result directories."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.axis_registry import MODEL_EVALUATION_AXES, PHYSICAL_PLAUSIBILITY, GEOMETRIC_INTEGRITY, APPLICATION_USEFULNESS
from scoring.aggregate import aggregate_sample_results, compute_sample_ranking_score
from scoring.report import generate_diagnostic_report, generate_report
from eval.metadata import build_run_metadata


SPECIAL = {"aggregate", "per_sample", "report", "run_metadata"}


def load_sample_results(shard_dirs: list[Path]) -> dict[str, dict]:
    results: dict[str, dict] = {}

    def quality(result: dict) -> tuple:
        validity = result.get("scoring_validity") or {}
        present = validity.get("present_axes") or []
        missing = validity.get("missing_required_axes") or []
        invalid = validity.get("invalid_judge_outputs") or []
        return (
            bool(result.get("scoring_complete")),
            result.get("sample_status") == "valid",
            not bool(result.get("skipped")),
            len(present),
            -len(missing),
            -len(invalid),
        )

    for shard_dir in shard_dirs:
        for path in shard_dir.glob("*.json"):
            if path.stem in SPECIAL:
                continue
            with path.open(encoding="utf-8") as f:
                result = json.load(f)
            task_id = result.get("task_id") or path.stem
            previous = results.get(task_id)
            # A transport failure in a later retry must never overwrite a
            # more complete earlier judgment. Equal-quality later retries do
            # replace earlier ones, preserving deterministic retry semantics.
            if previous is None or quality(result) >= quality(previous):
                results[task_id] = result
    return results


def align_results_to_manifest(results_by_id: dict[str, dict], task_ids: list[str]) -> list[dict]:
    """Keep missing tasks visible so a partial shard set cannot publish."""
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate task IDs in requested manifest")
    return [results_by_id.get(task_id, {
        "task_id": task_id,
        "skipped": True,
        "sample_status": "evaluator_invalid",
        "scoring_complete": False,
        "skip_reason": "missing_shard_result",
    }) for task_id in task_ids]


def load_shard_metadata(shard_dirs: list[Path]) -> list[dict]:
    metadata = []
    for shard_dir in shard_dirs:
        path = shard_dir / "run_metadata.json"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            metadata.append(json.load(f))
    return metadata


def aggregate_group(results: list[dict]) -> dict:
    aggregate = aggregate_sample_results(results)
    axis_means = aggregate.get("axis_scores", {})
    aggregate["num_samples"] = len(results)
    aggregate["low_fidelity_flags"] = {
        "physical_plausibility_low": axis_means.get(PHYSICAL_PLAUSIBILITY, 100.0) < 35.0,
        "geometric_integrity_low": axis_means.get(GEOMETRIC_INTEGRITY, 100.0) < 35.0,
    }
    return aggregate


def write_axis_details(path: Path, results: list[dict]) -> None:
    """Export every raw and capped public axis, retaining invalid rows for audit."""
    axes = list(MODEL_EVALUATION_AXES)
    fields = ["task_id", "sample_status", "scoring_complete", "observable_event_coverage",
              "ranking_score", "invalid_judge_outputs"]
    fields += [f"raw_{axis}" for axis in axes]
    fields += [f"capped_{axis}" for axis in axes]
    fields += ["constraint_axis_adjustments"]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            scored = result.get("scored") or {}
            raw = dict(scored.get("raw_axis_scores") or {})
            capped = dict(scored.get("axis_scores") or {})
            raw[APPLICATION_USEFULNESS] = scored.get("raw_application_usefulness_score")
            capped[APPLICATION_USEFULNESS] = scored.get("application_usefulness_score")
            row = {"task_id": result.get("task_id"), "sample_status": result.get("sample_status"),
                   "scoring_complete": result.get("scoring_complete"),
                   "observable_event_coverage": result.get("observable_event_coverage"),
                   "ranking_score": compute_sample_ranking_score(result) if result.get("scoring_complete") else None,
                   "invalid_judge_outputs": json.dumps((result.get("scoring_validity") or {}).get("invalid_judge_outputs", []), ensure_ascii=False),
                   "constraint_axis_adjustments": json.dumps(scored.get("constraint_axis_adjustments", {}), ensure_ascii=False)}
            row.update({f"raw_{axis}": raw.get(axis) for axis in axes})
            row.update({f"capped_{axis}": capped.get(axis) for axis in axes})
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("shard_dirs", nargs="+")
    args = parser.parse_args()

    shard_dirs = [Path(p) for p in args.shard_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_id = load_sample_results(shard_dirs)
    with open(args.samples_json, encoding="utf-8") as f:
        sample_order = [s["task_id"] for s in json.load(f)["samples"]]
    all_results = align_results_to_manifest(results_by_id, sample_order)

    for result in all_results:
        task_id = result["task_id"]
        with (output_dir / f"{task_id}.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

    aggregate = aggregate_sample_results(all_results)
    shard_metadata = load_shard_metadata(shard_dirs)
    base_metadata = shard_metadata[0] if shard_metadata else {}
    run_metadata = build_run_metadata(
        model_name=args.model,
        video_dir=base_metadata.get("video_dir", ""),
        samples_json=args.samples_json,
        output_dir=str(output_dir),
        llm_provider=base_metadata.get("llm_provider", "openai_compat"),
        use_llm=bool(base_metadata.get("llm_enabled", True)),
        model_answers_path=base_metadata.get("model_answers_path"),
    )
    run_metadata["combined_from_shards"] = [str(p) for p in shard_dirs]
    run_metadata["source_shard_metadata"] = shard_metadata
    aggregate["run_metadata"] = run_metadata
    completed = [r for r in all_results if not r.get("skipped")]
    if completed:
        aggregate["domain_breakdown"] = {
            domain: aggregate_group([r for r in completed if r.get("domain") == domain])
            for domain in sorted({r.get("domain") for r in completed})
        }
        aggregate["task_breakdown"] = {
            task: aggregate_group([r for r in completed if r.get("task_category") == task])
            for task in sorted({r.get("task_category") for r in completed})
        }
        aggregate["difficulty_breakdown"] = {
            level: aggregate_group([r for r in completed if r.get("difficulty_level") == level])
            for level in sorted({r.get("difficulty_level") for r in completed if r.get("difficulty_level")})
        }
        aggregate["model_evaluation_axes"] = MODEL_EVALUATION_AXES
        aggregate["low_fidelity_summary"] = {
            "domains_physical_low": [
                domain
                for domain, item in aggregate["domain_breakdown"].items()
                if item["low_fidelity_flags"]["physical_plausibility_low"]
            ],
            "domains_geometric_low": [
                domain
                for domain, item in aggregate["domain_breakdown"].items()
                if item["low_fidelity_flags"]["geometric_integrity_low"]
            ],
        }

    aggregate["combined_from_shards"] = [str(p) for p in shard_dirs]
    aggregate["num_unique_sample_files"] = len(results_by_id)

    with (output_dir / "per_sample.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    write_axis_details(output_dir / "axis_details.csv", all_results)
    with (output_dir / "aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, default=str)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, default=str)
    report = generate_report(generate_diagnostic_report(args.model, aggregate, all_results))
    with (output_dir / "report.json").open("w", encoding="utf-8") as f:
        f.write(report)

    print(json.dumps({
        "output_dir": str(output_dir),
        "num_results": len(all_results),
        "num_unique_sample_files": len(results_by_id),
        "completed": aggregate.get("num_samples_completed"),
        "skipped": aggregate.get("num_samples_skipped"),
        "overall": aggregate.get("overall"),
        "ranking_score": aggregate.get("ranking_score"),
    }, indent=2))


if __name__ == "__main__":
    main()
