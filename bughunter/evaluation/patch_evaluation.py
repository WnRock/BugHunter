"""
BugHunter - Patch evaluation utilities for fix patch results
"""

import os
import sys
import json
import logging
import tempfile
import subprocess
from typing import Dict, Any
from bughunter.config.manager import config_manager
from bughunter.core.docker_manager import DockerManager
from concurrent.futures import ProcessPoolExecutor, as_completed


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

            copy_result = subprocess.run(
                copy_cmd, shell=True, capture_output=True, text=True
            )

            if copy_result.returncode != 0:
                raise Exception(
                    f"Failed to copy patch to container: {copy_result.stderr}"
                )

            # Check which test script exists and make it executable
            script_names = ["fix_run.sh", "fix-run.sh"]
            test_script = None

            for script_name in script_names:
                check_result = docker_manager.execute_command(
                    f"test -f /home/{script_name}"
                )
                if check_result.exit_code == 0:
                    test_script = f"/home/{script_name}"
                    break

            if not test_script:
                raise Exception("Neither fix_run.sh nor fix-run.sh found in container")

            # Make the test script executable
            docker_manager.execute_command(f"chmod +x {test_script}")

            # Run the test script
            test_result = docker_manager.execute_command(test_script)

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


def run_patch_evaluation(args):
    """Run evaluation of fix patch results"""
    results_file = args.results

    # Check if results file exists
    if not os.path.exists(results_file):
        logging.error(f"Results file not found: {results_file}")
        sys.exit(1)

    # Load config to get number of workers
    try:
        config = config_manager.get_config()
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
