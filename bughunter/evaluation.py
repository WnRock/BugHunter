"""
BugHunter - Evaluation utilities for locate_bug task results
"""

import re
import os
import sys
import yaml
import json
import logging
import tempfile
from typing import Dict, Any, Optional
from bughunter.core.docker_manager import DockerManager
from concurrent.futures import ProcessPoolExecutor, as_completed


def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not os.path.isabs(config_file):
        config_file = os.path.join(os.path.dirname(__file__), "..", config_file)

    with open(config_file, "r") as f:
        return yaml.safe_load(f)


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


def evaluate_patch_results(results_file: str, num_workers: int = 1) -> Dict[str, Any]:
    """Evaluate fix patch results by running tests in Docker containers"""

    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    # Load results
    with open(results_file, "r") as f:
        results = json.load(f)

    # Filter results that have patches
    patch_results = []
    for result in results:
        if result.get("success", False) and result.get("result_data", {}).get("patch"):
            patch_results.append(result)

    if not patch_results:
        return {
            "total_instances": len(results),
            "patch_instances": 0,
            "evaluated_instances": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "error_instances": 0,
            "success_rate": 0.0,
            "detailed_results": [],
        }

    logging.info(f"Found {len(patch_results)} instances with patches to evaluate")

    detailed_results = []
    passed_tests = 0
    failed_tests = 0
    error_instances = 0

    # Use multiprocessing if num_workers > 1 and multiple instances
    if num_workers > 1 and len(patch_results) > 1:
        logging.info(
            f"Using parallel processing with {num_workers} workers for {len(patch_results)} instances"
        )

        # Prepare arguments for worker processes
        worker_args = []
        for result in patch_results:
            instance_id = result["instance_id"]
            patch_content = result["result_data"]["patch"]
            environment = None
            if "trajectory_summary" in result:
                environment = result["trajectory_summary"].get("environment")

            if environment:
                worker_args.append((environment, patch_content, instance_id))
            else:
                # Handle missing environment case
                error_instances += 1
                detailed_results.append(
                    {
                        "instance_id": instance_id,
                        "test_passed": False,
                        "error": "No environment information found",
                        "exit_code": None,
                    }
                )

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_instance = {
                executor.submit(worker_run_patch_test, args): args[
                    2
                ]  # args[2] is instance_id
                for args in worker_args
            }

            # Process results as they complete
            for future in as_completed(future_to_instance):
                instance_id = future_to_instance[future]
                try:
                    test_result = future.result()

                    if test_result["test_passed"]:
                        passed_tests += 1
                    else:
                        failed_tests += 1

                    detailed_results.append(test_result)

                except Exception as e:
                    logging.error(f"Error evaluating {instance_id}: {e}")
                    error_instances += 1
                    detailed_results.append(
                        {
                            "instance_id": instance_id,
                            "test_passed": False,
                            "error": str(e),
                            "exit_code": None,
                        }
                    )
    else:
        # Sequential processing (original logic)
        if num_workers > 1:
            logging.info("Only one instance to process, using sequential processing")
        else:
            logging.info(
                f"Using sequential processing for {len(patch_results)} instances"
            )

        for result in patch_results:
            instance_id = result["instance_id"]
            patch_content = result["result_data"]["patch"]

            # Extract environment from trajectory summary
            environment = None
            if "trajectory_summary" in result:
                environment = result["trajectory_summary"].get("environment")

            if not environment:
                logging.warning(
                    f"No environment found for instance {instance_id}, skipping"
                )
                error_instances += 1
                detailed_results.append(
                    {
                        "instance_id": instance_id,
                        "test_passed": False,
                        "error": "No environment information found",
                        "exit_code": None,
                    }
                )
                continue

            logging.info(
                f"Evaluating patch for {instance_id} using environment {environment}"
            )

            # Run patch evaluation
            try:
                test_result = run_patch_test(environment, patch_content, instance_id)

                if test_result["test_passed"]:
                    passed_tests += 1
                else:
                    failed_tests += 1

                detailed_results.append(test_result)

            except Exception as e:
                logging.error(f"Error evaluating {instance_id}: {e}")
                error_instances += 1
                detailed_results.append(
                    {
                        "instance_id": instance_id,
                        "test_passed": False,
                        "error": str(e),
                        "exit_code": None,
                    }
                )

    evaluated_instances = passed_tests + failed_tests
    success_rate = (
        passed_tests / evaluated_instances if evaluated_instances > 0 else 0.0
    )

    return {
        "total_instances": len(results),
        "patch_instances": len(patch_results),
        "evaluated_instances": evaluated_instances,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "error_instances": error_instances,
        "success_rate": success_rate,
        "detailed_results": detailed_results,
    }


def worker_run_patch_test(args):
    """
    Worker function to run patch test in a separate process.
    This function needs to be at module level for multiprocessing to work.
    """
    environment, patch_content, instance_id = args

    # Set up logging for this worker process
    logging.basicConfig(
        level=logging.INFO,
        format=f"[PatchWorker-{instance_id}] %(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    try:
        logging.info(
            f"Evaluating patch for {instance_id} using environment {environment}"
        )
        return run_patch_test(environment, patch_content, instance_id)
    except Exception as e:
        logging.error(f"Error in worker for {instance_id}: {e}")
        return {
            "instance_id": instance_id,
            "test_passed": False,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "error": str(e),
        }


def run_patch_test(
    environment: str, patch_content: str, instance_id: str
) -> Dict[str, Any]:
    """Run patch test in Docker container"""

    docker_manager = DockerManager(timeout=600)  # 10 minute timeout

    try:
        # Start container
        if not docker_manager.start_container(environment):
            raise Exception(f"Failed to start container for environment {environment}")

        # Create temporary patch file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".patch", delete=False
        ) as temp_patch:
            temp_patch.write(patch_content)
            temp_patch_path = temp_patch.name

        try:
            # Copy patch to container at /home/fix.patch
            copy_cmd = f"docker cp {temp_patch_path} {docker_manager.container.id}:/home/fix.patch"
            import subprocess

            copy_result = subprocess.run(
                copy_cmd, shell=True, capture_output=True, text=True
            )

            if copy_result.returncode != 0:
                raise Exception(
                    f"Failed to copy patch to container: {copy_result.stderr}"
                )

            # Make sure fix_run.sh is executable and run it
            docker_manager.execute_command("chmod +x /home/fix_run.sh")

            # Run the test script
            test_result = docker_manager.execute_command("/home/fix_run.sh")

            return {
                "instance_id": instance_id,
                "test_passed": test_result.exit_code == 0,
                "exit_code": test_result.exit_code,
                "stdout": test_result.stdout,
                "stderr": test_result.stderr,
                "error": None,
            }

        finally:
            # Clean up temporary patch file
            os.unlink(temp_patch_path)

    except Exception as e:
        return {
            "instance_id": instance_id,
            "test_passed": False,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "error": str(e),
        }

    finally:
        # Always stop the container
        docker_manager.stop_container()


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


def run_patch_evaluation(args):
    """Run evaluation of fix patch results"""
    results_file = args.results

    # Check if results file exists
    if not os.path.exists(results_file):
        logging.error(f"Results file not found: {results_file}")
        sys.exit(1)

    # Load config to get number of workers
    try:
        config = load_config()  # Use default config.yaml
        num_workers = config["system"].get("num_workers", 1)
    except Exception as e:
        logging.warning(
            f"Failed to load config for workers setting: {e}. Using 1 worker."
        )
        num_workers = 1

    logging.info(f"Evaluating patch results from: {results_file}")
    if num_workers > 1:
        logging.info(f"Using {num_workers} workers for parallel patch evaluation")

    # Run evaluation
    try:
        evaluation_results = evaluate_patch_results(results_file, num_workers)

        # Print summary
        print("\n" + "=" * 60)
        print("FIX PATCH EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total instances: {evaluation_results['total_instances']}")
        print(f"Instances with patches: {evaluation_results['patch_instances']}")
        print(f"Evaluated instances: {evaluation_results['evaluated_instances']}")
        print(f"Error instances: {evaluation_results['error_instances']}")
        print(f"Tests passed: {evaluation_results['passed_tests']}")
        print(f"Tests failed: {evaluation_results['failed_tests']}")
        print(f"Success rate: {evaluation_results['success_rate']:.2%}")
        print("=" * 60)

        # Print detailed results if requested
        if args.detailed:
            print("\nDETAILED RESULTS:")
            print("-" * 60)
            for detail in evaluation_results["detailed_results"]:
                if detail.get("error"):
                    print(f"ERROR: {detail['instance_id']} - {detail['error']}")
                else:
                    status = "✓ PASSED" if detail["test_passed"] else "✗ FAILED"
                    print(
                        f"{status}: {detail['instance_id']} (exit code: {detail['exit_code']})"
                    )
                    if detail.get("stdout"):
                        print(
                            f"  STDOUT: {detail['stdout'][:200]}{'...' if len(detail['stdout']) > 200 else ''}"
                        )
                    if detail.get("stderr"):
                        print(
                            f"  STDERR: {detail['stderr'][:200]}{'...' if len(detail['stderr']) > 200 else ''}"
                        )
                print("-" * 40)

        # Save evaluation results if output file specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(evaluation_results, f, indent=2)
            print(f"\nDetailed evaluation results saved to: {args.output}")

    except Exception as e:
        logging.error(f"Patch evaluation failed: {e}")
        sys.exit(1)
