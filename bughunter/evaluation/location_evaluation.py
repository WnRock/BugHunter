"""
BugHunter - Location evaluation utilities for locate_bug task results
"""

import re
import os
import sys
import yaml
import json
import logging
from typing import Dict, Any, List


def load_gold_truth_data(test_data_file: str) -> Dict[str, Any]:
    """Load gold truth data from test dataset"""
    if not os.path.isabs(test_data_file):
        test_data_file = os.path.join(os.path.dirname(__file__), "..", test_data_file)

    with open(test_data_file, "r") as f:
        data = yaml.safe_load(f)

    gold_truth = {}
    for item in data:
        instance_id = item["instance_id"]
        gold_target_file = item.get("gold_target_file")
        gold_truth[instance_id] = gold_target_file

    return gold_truth


def normalize_path(path: str) -> str:
    """Normalize file path by removing /home/repo prefix and converting to relative path"""
    if not path:
        return ""

    # Remove leading /home/ patterns and repo name
    path = str(path).strip()

    # Match patterns like /home/repo_name/... or /home/any_name/...
    home_pattern = r"^/home/[^/]+/"
    if re.match(home_pattern, path):
        path = re.sub(home_pattern, "", path)

    # Remove leading slash if present
    if path.startswith("/"):
        path = path[1:]

    return path


def extract_location_candidates_from_result(result_data: Dict[str, Any]) -> List[str]:
    """Extract location candidates from task result data"""
    candidates = []

    # Get location candidates from new format
    if "location_candidates" in result_data:
        location_candidates = result_data["location_candidates"]
        if isinstance(location_candidates, list):
            for candidate in location_candidates:
                if isinstance(candidate, dict) and "file_path" in candidate:
                    file_path = candidate["file_path"]
                    if file_path:
                        candidates.append(normalize_path(file_path))

    return candidates


def check_location_match(predicted_path: str, gold_target_file: Any) -> bool:
    """Check if a predicted path matches the gold target file"""
    if not predicted_path:
        return False

    if isinstance(gold_target_file, list):
        # Check if prediction matches any file in the list
        for gold_file in gold_target_file:
            normalized_gold = normalize_path(str(gold_file))
            if predicted_path == normalized_gold or normalized_gold in predicted_path:
                return True
        return False

    elif isinstance(gold_target_file, str):
        # Check if prediction contains the gold target file
        normalized_gold = normalize_path(gold_target_file)
        return normalized_gold in predicted_path or predicted_path == normalized_gold

    return False


def calculate_pass_at_k_metrics(
    location_candidates: List[str],
    gold_target_file: Any,
    k_values: List[int] = [1, 2, 3, 5],
) -> Dict[str, bool]:
    """Calculate Pass@k metrics for location candidates"""
    metrics = {}

    for k in k_values:
        # Consider the top k candidates
        top_k_candidates = location_candidates[:k]

        # Check if any of the top k candidates match the gold target
        pass_at_k = any(
            check_location_match(candidate, gold_target_file)
            for candidate in top_k_candidates
        )
        metrics[f"pass_at_{k}"] = pass_at_k

    return metrics


def evaluate_location_candidates(
    result_data: Dict[str, Any], gold_target_file: Any
) -> Dict[str, Any]:
    """
    Evaluate location candidates with Pass@k metrics

    Args:
        result_data: Task result data containing location candidates
        gold_target_file: Ground truth - can be None, string, or list

    Returns:
        Dict with evaluation results including Pass@k metrics
    """
    result = {
        "gold_target": gold_target_file,
        "location_candidates": [],
        "pass_at_k": {},
        "best_match_rank": None,
        "reason": "",
    }

    # Skip evaluation if no gold target file
    if gold_target_file is None:
        result["reason"] = "No gold target file to compare against"
        return result

    # Extract location candidates
    location_candidates = extract_location_candidates_from_result(result_data)
    result["location_candidates"] = location_candidates

    if not location_candidates:
        result["reason"] = "No location candidates found in result data"
        return result

    # Calculate Pass@k metrics
    pass_at_k_metrics = calculate_pass_at_k_metrics(
        location_candidates, gold_target_file
    )
    result["pass_at_k"] = pass_at_k_metrics

    # Find the rank of the best match
    for i, candidate in enumerate(location_candidates):
        if check_location_match(candidate, gold_target_file):
            result["best_match_rank"] = i + 1  # 1-indexed rank
            break

    if result["best_match_rank"] is not None:
        result["reason"] = f"Correct location found at rank {result['best_match_rank']}"
    else:
        result["reason"] = (
            f"No correct location found in {len(location_candidates)} candidates"
        )

    return result


def evaluate_locate_bug_results(
    results_file: str, test_data_file: str
) -> Dict[str, Any]:
    """
    Evaluate locate_bug results against gold truth data with Pass@k metrics

    Args:
        results_file: Path to results.json file with LLM outputs
        test_data_file: Path to test_set.yaml file with gold truth

    Returns:
        Dict with evaluation statistics and detailed results
    """
    # Load results and gold truth
    with open(results_file, "r") as f:
        results = json.load(f)

    gold_truth = load_gold_truth_data(test_data_file)

    evaluation_results = {
        "total_instances": 0,
        "evaluated_instances": 0,
        "skipped_instances": 0,
        "pass_at_k_summary": {
            "pass_at_1": {"correct": 0, "total": 0},
            "pass_at_2": {"correct": 0, "total": 0},
            "pass_at_3": {"correct": 0, "total": 0},
            "pass_at_5": {"correct": 0, "total": 0},
        },
        "average_best_rank": 0.0,
        "detailed_results": [],
    }

    best_ranks = []

    for result in results:
        instance_id = result["instance_id"]
        task_type = result["task_type"]

        # Only evaluate locate_bug tasks
        if task_type != "locate_bug":
            continue

        evaluation_results["total_instances"] += 1

        # Get gold truth for this instance
        gold_target_file = gold_truth.get(instance_id)

        # Skip if no gold target file
        if gold_target_file is None:
            evaluation_results["skipped_instances"] += 1
            evaluation_results["detailed_results"].append(
                {
                    "instance_id": instance_id,
                    "skipped": True,
                    "reason": "No gold target file",
                }
            )
            continue

        evaluation_results["evaluated_instances"] += 1

        # Extract result data
        result_data = result.get("result_data", {})

        # Evaluate with Pass@k metrics
        eval_result = evaluate_location_candidates(result_data, gold_target_file)

        # Update Pass@k statistics
        for k in [1, 2, 3, 5]:
            metric_key = f"pass_at_{k}"
            if metric_key in eval_result["pass_at_k"]:
                evaluation_results["pass_at_k_summary"][metric_key]["total"] += 1
                if eval_result["pass_at_k"][metric_key]:
                    evaluation_results["pass_at_k_summary"][metric_key]["correct"] += 1

        # Track best ranks for average calculation
        if eval_result["best_match_rank"] is not None:
            best_ranks.append(eval_result["best_match_rank"])

        # Store detailed result
        detailed_result = {
            "instance_id": instance_id,
            "location_candidates": eval_result["location_candidates"],
            "pass_at_k": eval_result["pass_at_k"],
            "best_match_rank": eval_result["best_match_rank"],
            "gold_target": eval_result["gold_target"],
            "reason": eval_result["reason"],
            "llm_success": result.get("success", False),
        }
        evaluation_results["detailed_results"].append(detailed_result)

    # Calculate Pass@k percentages
    for k in [1, 2, 3, 5]:
        metric_key = f"pass_at_{k}"
        summary = evaluation_results["pass_at_k_summary"][metric_key]
        if summary["total"] > 0:
            summary["percentage"] = summary["correct"] / summary["total"]
        else:
            summary["percentage"] = 0.0

    # Calculate average best rank
    if best_ranks:
        evaluation_results["average_best_rank"] = sum(best_ranks) / len(best_ranks)

    return evaluation_results


def run_evaluation(args):
    """Run evaluation of locate_bug results with Pass@k metrics"""
    results_file = args.results
    test_data_file = args.test_data or "data/test_set.yaml"

    # Check if files exist
    if not os.path.exists(results_file):
        logging.error(f"Results file not found: {results_file}")
        sys.exit(1)

    if not os.path.isabs(test_data_file):
        test_data_file = os.path.join(os.path.dirname(__file__), "..", test_data_file)

    if not os.path.exists(test_data_file):
        logging.error(f"Test data file not found: {test_data_file}")
        sys.exit(1)

    logging.info(f"Evaluating results from: {results_file}")
    logging.info(f"Using test data from: {test_data_file}")

    # Run evaluation
    try:
        evaluation_results = evaluate_locate_bug_results(results_file, test_data_file)

        # Print summary
        print("\n" + "=" * 60)
        print("LOCATE BUG EVALUATION RESULTS WITH PASS@K METRICS")
        print("=" * 60)
        print(f"Total locate_bug instances: {evaluation_results['total_instances']}")
        print(f"Evaluated instances: {evaluation_results['evaluated_instances']}")
        print(
            f"Skipped instances (no gold target): {evaluation_results['skipped_instances']}"
        )
        print()
        print("Pass@k Metrics:")
        for k in [1, 2, 3, 5]:
            metric_key = f"pass_at_{k}"
            summary = evaluation_results["pass_at_k_summary"][metric_key]
            percentage = summary.get("percentage", 0.0)
            print(
                f"  Pass@{k}: {summary['correct']}/{summary['total']} ({percentage:.2%})"
            )

        if evaluation_results["average_best_rank"] > 0:
            print(
                f"\nAverage rank of correct location: {evaluation_results['average_best_rank']:.2f}"
            )
        print("=" * 60)

        # Print detailed results if requested
        if args.detailed:
            print("\nDETAILED RESULTS:")
            print("-" * 60)
            for detail in evaluation_results["detailed_results"]:
                if detail.get("skipped"):
                    print(f"SKIPPED: {detail['instance_id']} - {detail['reason']}")
                else:
                    pass_at_1 = (
                        "✓" if detail["pass_at_k"].get("pass_at_1", False) else "✗"
                    )
                    rank_info = (
                        f" (rank {detail['best_match_rank']})"
                        if detail["best_match_rank"]
                        else " (not found)"
                    )
                    print(f"{pass_at_1} {detail['instance_id']}{rank_info}")
                    print(f"  Candidates: {len(detail['location_candidates'])}")
                    print(f"  Pass@1: {detail['pass_at_k'].get('pass_at_1', False)}")
                    print(f"  Pass@3: {detail['pass_at_k'].get('pass_at_3', False)}")
                    print(f"  Pass@5: {detail['pass_at_k'].get('pass_at_5', False)}")
                    print(f"  Gold Target: {detail['gold_target']}")
                    print(f"  LLM Success: {detail['llm_success']}")
                    print("-" * 40)

        # Save evaluation results if output file specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(evaluation_results, f, indent=2)
            print(f"\nDetailed evaluation results saved to: {args.output}")

    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)
