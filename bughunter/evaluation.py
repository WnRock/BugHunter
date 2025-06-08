"""
BugHunter - Evaluation utilities for locate_bug task results
"""

import re
import os
import sys
import yaml
import json
import logging
from typing import Dict, Any, Optional


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


def extract_file_path_from_llm_output(llm_output: str) -> Optional[str]:
    """Extract file path from LLM output for locate_bug task"""
    if not llm_output or not isinstance(llm_output, str):
        return None

    # Look for LOCATION_FOUND pattern first
    location_pattern = r"LOCATION_FOUND:\s*([^:\s\n]+)"
    match = re.search(location_pattern, llm_output, re.IGNORECASE)
    if match:
        return normalize_path(match.group(1))

    # Look for file paths in common formats
    file_patterns = [
        r"(/[\w/\.-]+\.[a-zA-Z]+)",  # Absolute paths like /path/to/file.ext
        r"([\w/\.-]+\.[a-zA-Z]+)",  # Relative paths like src/file.ext
    ]

    for pattern in file_patterns:
        matches = re.findall(pattern, llm_output)
        if matches:
            # Return the first reasonable match (filter out very short matches)
            for match in matches:
                normalized = normalize_path(match)
                if len(normalized) > 3 and "." in normalized:  # Basic sanity check
                    return normalized

    return None


def evaluate_location_correctness(
    llm_output: str, gold_target_file: Any
) -> Dict[str, Any]:
    """
    Evaluate if LLM's location output matches the gold target file

    Args:
        llm_output: The LLM's output text containing location information
        gold_target_file: Ground truth - can be None, string, or list

    Returns:
        Dict with evaluation results
    """
    result = {
        "correct": False,
        "llm_prediction": None,
        "gold_target": gold_target_file,
        "reason": "",
    }

    # Skip evaluation if no gold target file
    if gold_target_file is None:
        result["reason"] = "No gold target file to compare against"
        return result

    # Extract file path from LLM output
    llm_file_path = extract_file_path_from_llm_output(llm_output)
    result["llm_prediction"] = llm_file_path

    if not llm_file_path:
        result["reason"] = "Could not extract file path from LLM output"
        return result

    # Handle different gold_target_file formats
    if isinstance(gold_target_file, list):
        # Check if LLM prediction matches any file in the list
        for gold_file in gold_target_file:
            normalized_gold = normalize_path(str(gold_file))
            if llm_file_path == normalized_gold or normalized_gold in llm_file_path:
                result["correct"] = True
                result["reason"] = (
                    f"LLM prediction matches gold file: {normalized_gold}"
                )
                return result
        result["reason"] = (
            f"LLM prediction '{llm_file_path}' not found in gold list: {gold_target_file}"
        )

    elif isinstance(gold_target_file, str):
        # Check if LLM prediction contains the gold target file
        normalized_gold = normalize_path(gold_target_file)
        if normalized_gold in llm_file_path or llm_file_path == normalized_gold:
            result["correct"] = True
            result["reason"] = f"LLM prediction contains gold file: {normalized_gold}"
        else:
            result["reason"] = (
                f"LLM prediction '{llm_file_path}' does not contain gold file: {normalized_gold}"
            )

    else:
        result["reason"] = (
            f"Unsupported gold_target_file type: {type(gold_target_file)}"
        )

    return result


def evaluate_locate_bug_results(
    results_file: str, test_data_file: str
) -> Dict[str, Any]:
    """
    Evaluate locate_bug results against gold truth data

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
        "correct_predictions": 0,
        "incorrect_predictions": 0,
        "accuracy": 0.0,
        "detailed_results": [],
    }

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

        # Extract LLM output
        llm_output = ""
        if result.get("success") and result.get("result_data"):
            # Try to get the full response from location task
            if "full_response" in result["result_data"]:
                llm_output = result["result_data"]["full_response"]
            elif "location" in result["result_data"]:
                # Fallback to location data
                location_data = result["result_data"]["location"]
                if isinstance(location_data, dict) and "file_path" in location_data:
                    llm_output = location_data["file_path"] or ""

        # Evaluate correctness
        eval_result = evaluate_location_correctness(llm_output, gold_target_file)

        # Update statistics
        if eval_result["correct"]:
            evaluation_results["correct_predictions"] += 1
        else:
            evaluation_results["incorrect_predictions"] += 1

        # Store detailed result
        detailed_result = {
            "instance_id": instance_id,
            "correct": eval_result["correct"],
            "llm_prediction": eval_result["llm_prediction"],
            "gold_target": eval_result["gold_target"],
            "reason": eval_result["reason"],
            "llm_success": result.get("success", False),
        }
        evaluation_results["detailed_results"].append(detailed_result)

    # Calculate accuracy
    if evaluation_results["evaluated_instances"] > 0:
        evaluation_results["accuracy"] = (
            evaluation_results["correct_predictions"]
            / evaluation_results["evaluated_instances"]
        )

    return evaluation_results


def run_evaluation(args):
    """Run evaluation of locate_bug results"""
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
        print("LOCATE BUG EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total locate_bug instances: {evaluation_results['total_instances']}")
        print(f"Evaluated instances: {evaluation_results['evaluated_instances']}")
        print(
            f"Skipped instances (no gold target): {evaluation_results['skipped_instances']}"
        )
        print(f"Correct predictions: {evaluation_results['correct_predictions']}")
        print(f"Incorrect predictions: {evaluation_results['incorrect_predictions']}")
        print(f"Accuracy: {evaluation_results['accuracy']:.2%}")
        print("=" * 60)

        # Print detailed results if requested
        if args.detailed:
            print("\nDETAILED RESULTS:")
            print("-" * 60)
            for detail in evaluation_results["detailed_results"]:
                if detail.get("skipped"):
                    print(f"SKIPPED: {detail['instance_id']} - {detail['reason']}")
                else:
                    status = "✓ CORRECT" if detail["correct"] else "✗ INCORRECT"
                    print(f"{status}: {detail['instance_id']}")
                    print(f"  LLM Prediction: {detail['llm_prediction']}")
                    print(f"  Gold Target: {detail['gold_target']}")
                    print(f"  Reason: {detail['reason']}")
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
