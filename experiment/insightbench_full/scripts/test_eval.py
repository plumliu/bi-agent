#!/usr/bin/env python3
"""
Use Llama (via OpenRouter) as the only judge model while reusing Insight-Bench
G-Eval prompts and one-to-many aggregation logic.

Input:
- result dir: generated prediction files, e.g. flag-1.json ... flag-6.json
- groundtruth dir: GT files with insights + summary

Output:
- eval_results/detailed_scores.json
- eval_results/summary_scores.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from eval_prompts import get_g_eval_prompt


@dataclass
class JudgeConfig:
    model: str
    max_retries: int
    retry_sleep_sec: float
    max_workers_flags: int = 16
    max_workers_insights: int = 16




def parse_rating(text: str) -> int:
    """
    Parse integer rating from response text.
    Preferred format: <rating>N</rating>
    Fallback: first integer in [1, 10].
    """
    tag_match = re.search(r"<rating>\s*(\d{1,2})\s*</rating>", text, flags=re.IGNORECASE)
    if tag_match:
        value = int(tag_match.group(1))
        if 1 <= value <= 10:
            return value

    # fallback: first integer 1..10
    for match in re.finditer(r"\b(10|[1-9])\b", text):
        value = int(match.group(1))
        if 1 <= value <= 10:
            return value

    raise ValueError(f"Cannot parse rating from judge output: {text[:300]}")


def judge_pair_score(
    client: OpenAI,
    prompt_template: str,
    system_message: str,
    pred_text: str,
    gt_text: str,
    cfg: JudgeConfig,
) -> float:
    user_prompt = prompt_template.format(answer=pred_text, gt_answer=gt_text)

    last_error = ""
    for attempt in range(1, cfg.max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=64,
            )
            content = resp.choices[0].message.content or ""
            rating_1_to_10 = parse_rating(content)
            return rating_1_to_10 / 10.0
        except Exception as e:
            last_error = str(e)
            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_sleep_sec)
                continue
            raise RuntimeError(f"Judge failed after {cfg.max_retries} attempts: {last_error}") from e

    raise RuntimeError("Unexpected judge loop exit")


def score_insights_o2m(
    client: OpenAI,
    prompt_template: str,
    system_message: str,
    pred_insights: List[str],
    gt_insights: List[str],
    cfg: JudgeConfig,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    One-to-many aggregation (Insight-Bench style):
    for each GT insight, score against all predicted insights and keep best match.
    Pred-GT pairs are evaluated concurrently.
    """
    if not gt_insights:
        return 0.0, []
    if not pred_insights:
        details = [
            {
                "gt_insight": gt,
                "best_pred_insight": "",
                "best_score": 0.0,
            }
            for gt in gt_insights
        ]
        return 0.0, details

    def best_for_gt(gt: str) -> Tuple[str, float, str]:
        """Return (gt, best_score, best_pred) for one GT insight."""
        def score_one(pred: str) -> float:
            return judge_pair_score(
                client=client,
                prompt_template=prompt_template,
                system_message=system_message,
                pred_text=pred,
                gt_text=gt,
                cfg=cfg,
            )

        with ThreadPoolExecutor(max_workers=cfg.max_workers_insights) as pool:
            futures = {pool.submit(score_one, pred): pred for pred in pred_insights}
            local_best_score = -1.0
            local_best_pred = ""
            for fut in as_completed(futures):
                pred = futures[fut]
                score = fut.result()
                if score > local_best_score:
                    local_best_score = score
                    local_best_pred = pred

        return gt, max(local_best_score, 0.0), local_best_pred

    details: List[Dict[str, Any]] = []
    best_scores: List[float] = []

    with ThreadPoolExecutor(max_workers=min(len(gt_insights), cfg.max_workers_insights)) as pool:
        futures = [pool.submit(best_for_gt, gt) for gt in gt_insights]
        for fut in as_completed(futures):
            gt, best_score, best_pred = fut.result()
            best_scores.append(best_score)
            details.append(
                {
                    "gt_insight": gt,
                    "best_pred_insight": best_pred,
                    "best_score": best_score,
                }
            )

    overall = sum(best_scores) / len(best_scores)
    return overall, details


def score_summary(
    client: OpenAI,
    prompt_template: str,
    system_message: str,
    pred_summary: str,
    gt_summary: str,
    cfg: JudgeConfig,
) -> float:
    if not pred_summary.strip() or not gt_summary.strip():
        return 0.0
    return judge_pair_score(
        client=client,
        prompt_template=prompt_template,
        system_message=system_message,
        pred_text=pred_summary,
        gt_text=gt_summary,
        cfg=cfg,
    )


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_flags(
    client: OpenAI,
    prompt_template: str,
    system_message: str,
    result_dir: Path,
    gt_dir: Path,
    start: int,
    end: int,
    cfg: JudgeConfig,
    insights_weight: float,
    sample_ids: List[int] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

    if sample_ids is None:
        sample_iter = list(range(start, end + 1))
    else:
        sample_iter = sample_ids

    def evaluate_one_flag(i: int) -> Dict[str, Any]:
        """Evaluate a single flag file."""
        flag_name = f"flag-{i}"
        result_path = result_dir / f"{flag_name}.json"
        gt_path = gt_dir / f"{flag_name}.json"

        row: Dict[str, Any] = {
            "flag": flag_name,
            "result_file": str(result_path),
            "groundtruth_file": str(gt_path),
            "insights_score": 0.0,
            "summary_score": 0.0,
            "combined_score": 0.0,
            "insight_matches": [],
            "error": None,
        }

        try:
            if not result_path.exists():
                raise FileNotFoundError(f"Missing result file: {result_path}")
            if not gt_path.exists():
                raise FileNotFoundError(f"Missing groundtruth file: {gt_path}")

            pred = load_json(result_path)
            gt = load_json(gt_path)

            pred_insights = pred.get("insights", [])
            pred_summary = pred.get("summary", "")

            gt_insight_list = gt.get("insight_list", [])
            gt_insights = [item.get("insight", "") for item in gt_insight_list if isinstance(item, dict)]
            gt_summary = gt.get("summary", "")

            if not isinstance(pred_insights, list):
                pred_insights = []
            if not isinstance(gt_insights, list):
                gt_insights = []
            if not isinstance(pred_summary, str):
                pred_summary = ""
            if not isinstance(gt_summary, str):
                gt_summary = ""

            insights_score, insight_matches = score_insights_o2m(
                client=client,
                prompt_template=prompt_template,
                system_message=system_message,
                pred_insights=pred_insights,
                gt_insights=gt_insights,
                cfg=cfg,
            )
            summary_score = score_summary(
                client=client,
                prompt_template=prompt_template,
                system_message=system_message,
                pred_summary=pred_summary,
                gt_summary=gt_summary,
                cfg=cfg,
            )

            combined = insights_weight * insights_score + (1.0 - insights_weight) * summary_score

            row["insights_score"] = insights_score
            row["summary_score"] = summary_score
            row["combined_score"] = combined
            row["insight_matches"] = insight_matches
        except Exception as e:
            row["error"] = str(e)

        return row

    all_rows: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=cfg.max_workers_flags) as pool:
        futures = [pool.submit(evaluate_one_flag, i) for i in sample_iter]
        for fut in as_completed(futures):
            row = fut.result()
            all_rows.append(row)
            print(f"✓ Completed {row['flag']}")

    ok_rows = [r for r in all_rows if not r.get("error")]
    fail_rows = [r for r in all_rows if r.get("error")]

    def safe_mean(values: List[float]) -> float:
        return (sum(values) / len(values)) if values else 0.0

    summary = {
        "num_total": len(all_rows),
        "num_success": len(ok_rows),
        "num_failed": len(fail_rows),
        "failed_flags": [r["flag"] for r in fail_rows],
        "mean_insights_score": safe_mean([r["insights_score"] for r in ok_rows]),
        "mean_summary_score": safe_mean([r["summary_score"] for r in ok_rows]),
        "mean_combined_score": safe_mean([r["combined_score"] for r in ok_rows]),
        "insights_weight": insights_weight,
        "summary_weight": 1.0 - insights_weight,
    }

    return all_rows, summary


def main():
    """
    Main evaluation function.
    Compares test results in experiment/insightbench_full/result/flag-*.json
    against ground truth in experiment/insightbench_full/data/flag-*.json
    """
    # Setup paths
    project_root = Path(__file__).resolve().parents[3]
    result_dir = project_root / "experiment/insightbench_full/result2"
    gt_dir = project_root / "experiment/insightbench_full/data"
    output_dir = project_root / "experiment/insightbench_full/eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get OpenRouter API key from environment
    api_key = "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"

    # Initialize client and config
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    cfg = JudgeConfig(
        model="meta-llama/llama-3.3-70b-instruct",
        max_retries=3,
        retry_sleep_sec=2.0,
    )

    # Load G-Eval prompts from eval_prompts.py
    prompt_template, system_message = get_g_eval_prompt()

    # Find all result files
    result_files = sorted(result_dir.glob("flag-*.json"))
    if not result_files:
        raise RuntimeError(f"No result files found in {result_dir}")

    # Extract sample IDs from result files
    sample_ids = []
    for f in result_files:
        match = re.match(r"flag-(\d+)\.json", f.name)
        if match:
            sample_ids.append(int(match.group(1)))

    if not sample_ids:
        raise RuntimeError("No valid flag-*.json files found")

    sample_ids = sorted(set(sample_ids))

    start_id = min(sample_ids)
    end_id = max(sample_ids)

    print("=" * 80)
    print("InsightBench Evaluation using Llama-3.3-70B-Instruct")
    print("=" * 80)
    print(f"Result directory: {result_dir}")
    print(f"Ground truth directory: {gt_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Sample range: {start_id} to {end_id}")
    print(f"Total samples found: {len(sample_ids)}")
    print(f"Judge model: {cfg.model}")
    print(f"Insights weight: 0.7, Summary weight: 0.3")
    print("=" * 80)
    print()

    # Run evaluation
    print("Starting evaluation...")
    all_rows, summary = evaluate_flags(
        client=client,
        prompt_template=prompt_template,
        system_message=system_message,
        result_dir=result_dir,
        gt_dir=gt_dir,
        start=start_id,
        end=end_id,
        cfg=cfg,
        insights_weight=0.7,  # 70% weight on insights, 30% on summary
        sample_ids=sample_ids,
    )

    # Save detailed results
    detailed_path = output_dir / "detailed_scores.json"
    with detailed_path.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    print(f"✓ Detailed scores saved to: {detailed_path}")

    # Save summary results
    summary_path = output_dir / "summary_scores.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✓ Summary scores saved to: {summary_path}")

    # Print summary
    print()
    print("=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Total samples: {summary['num_total']}")
    print(f"Successful: {summary['num_success']}")
    print(f"Failed: {summary['num_failed']}")
    if summary['failed_flags']:
        print(f"Failed flags: {', '.join(summary['failed_flags'])}")
    print()
    print(f"Mean Insights Score: {summary['mean_insights_score']:.4f}")
    print(f"Mean Summary Score: {summary['mean_summary_score']:.4f}")
    print(f"Mean Combined Score: {summary['mean_combined_score']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
