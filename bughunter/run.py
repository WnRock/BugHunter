"""
BugHunter - Main Pipeline
"""

import os
import sys
import yaml
import json
import time
import logging
from typing import Dict, Any, List
from bughunter.config.manager import config_manager
from bughunter.tasks.fix_bug_task import FixBugTask
from bughunter.utils.setup import main as setup_main
from bughunter.tasks.locate_bug_task import LocateBugTask
from concurrent.futures import ProcessPoolExecutor, as_completed
from bughunter.tasks.fix_with_location_task import FixWithLocationTask
from bughunter.config.utils import create_argument_parser, handle_overrides
from bughunter.core.models import TestInstance, TaskResult, TaskType, AgentConfig
from bughunter.utils.output_manager import (
    save_instance_result_files,
    save_trajectory_file,
    save_fixed_output_files,
)


class IssueSolvingPipeline:
    """Main pipeline for solving issues with different task types"""

    def __init__(
        self,
        agent_config: AgentConfig,
        prompts_config: dict = None,
        model_config: dict = None,
    ):
        self.agent_config = agent_config
        self.prompts_config = prompts_config
        self.model_config = model_config
        self.tasks = {
            TaskType.FIX_BUG: FixBugTask(agent_config, prompts_config, model_config),
            TaskType.LOCATE_BUG: LocateBugTask(
                agent_config, prompts_config, model_config
            ),
            TaskType.FIX_WITH_LOCATION: FixWithLocationTask(
                agent_config, prompts_config, model_config
            ),
        }

    def solve_issue(self, test_instance: TestInstance) -> TaskResult:
        """Solve a single issue using the appropriate task type"""
        task = self.tasks[test_instance.task_type]
        return task.execute(test_instance)


def load_test_data() -> List[TestInstance]:
    """Load test instances from YAML file specified in config"""
    yaml_file = config_manager.get("tasks.test_data_file")

    # Adjust path to be relative to project root
    if not os.path.isabs(yaml_file):
        yaml_file = os.path.join(os.path.dirname(__file__), "..", yaml_file)

    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    instances = []

    # Use task type from config with validation
    try:
        task_type = TaskType(config_manager.get("tasks.task_type"))
    except ValueError as e:
        raise ValueError(
            f'Invalid task_type in config: {config_manager.get("tasks.task_type")}. Must be one of: {[t.value for t in TaskType]}'
        )

    # Optional instance filtering from config
    target_instances = config_manager.get("tasks.target_instances")

    for item in data:
        # Skip if target_instances is specified and this instance is not in the list
        if target_instances and item["instance_id"] not in target_instances:
            continue

        instances.append(
            TestInstance(
                image_name=item["image_name"],
                instance_id=item["instance_id"],
                problem_statement=item["problem_statement"],
                task_type=task_type,
                gold_target_file=item.get("gold_target_file"),
            )
        )

    if not instances:
        if target_instances:
            raise ValueError(
                f"No instances found matching target_instances: {target_instances}"
            )
        else:
            raise ValueError("No test instances found in the test data file")

    return instances


def setup_logging():
    """Setup logging configuration from config"""
    log_level = config_manager.get("system.log_level", "INFO")

    # Get output directory for log files
    base_output_dir = config_manager.get("output.output_dir")
    if base_output_dir is None:
        base_output_dir = os.getcwd()
    elif not os.path.isabs(base_output_dir):
        base_output_dir = os.path.abspath(base_output_dir)

    # Create output directory if it doesn't exist
    os.makedirs(base_output_dir, exist_ok=True)

    # Setup main pipeline logger with both console and file output
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(console_handler)

    # File handler for main pipeline log
    main_log_file = os.path.join(base_output_dir, "bughunter_pipeline.log")
    file_handler = logging.FileHandler(
        main_log_file, mode="w"
    )  # Overwrite previous log
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    logger.addHandler(file_handler)


def convert_result_to_dict(result: TaskResult) -> Dict[str, Any]:
    """Convert TaskResult to dictionary for JSON serialization"""
    result_dict = {
        "success": result.success,
        "instance_id": result.instance_id,
        "task_type": result.task_type.value,
        "iterations": result.iterations,
        "result_data": result.result_data,
        "error": result.error,
    }

    # Include trajectory if available
    if result.trajectory:
        result_dict["trajectory"] = result.trajectory.to_dict()

    return result_dict


def create_trajectory_summary(trajectory_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of trajectory for performance optimization"""
    return {
        "environment": trajectory_dict["environment"],
        "steps_count": len(trajectory_dict["trajectory"]),
        "history_count": len(trajectory_dict["history"]),
        "exit_status": (
            trajectory_dict["info"]["exit_status"]
            if trajectory_dict["info"]
            else "unknown"
        ),
    }


def create_error_result(
    instance_id: str, task_type: TaskType, error_msg: str
) -> Dict[str, Any]:
    """Create a standardized error result dictionary"""
    return {
        "success": False,
        "instance_id": instance_id,
        "task_type": task_type.value,
        "iterations": 0,
        "result_data": {},
        "error": error_msg,
    }


def process_result_for_output(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Process result dictionary for output, replacing full trajectory with summary"""
    if "trajectory" in result_dict:
        result_dict["trajectory_summary"] = create_trajectory_summary(
            result_dict["trajectory"]
        )
        del result_dict["trajectory"]
    return result_dict


def process_single_instance(
    test_instance: TestInstance, pipeline: IssueSolvingPipeline, output_dir: str
) -> Dict[str, Any]:
    """Process a single test instance and return result dictionary"""
    logging.info(
        f"Processing {test_instance.task_type.value} task: {test_instance.instance_id}"
    )

    try:
        result = pipeline.solve_issue(test_instance)

        # Save instance-specific result files
        save_instance_result_files(result, output_dir)
        save_trajectory_file(result, output_dir)

        # Convert result and process for output
        result_dict = convert_result_to_dict(result)
        return process_result_for_output(result_dict)

    except Exception as e:
        logging.error(f"Error processing instance {test_instance.instance_id}: {e}")
        return create_error_result(
            test_instance.instance_id, test_instance.task_type, str(e)
        )


def save_intermediate_results(results: List[Dict[str, Any]], output_path: str):
    """Save intermediate results to file"""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def print_results_summary(results: List[Dict[str, Any]], output_dir: str):
    """Print summary of results"""
    successful = len([r for r in results if r["success"]])
    total = len(results)

    # Task type breakdown
    task_summary = {}
    for result in results:
        task_type = result["task_type"]
        if task_type not in task_summary:
            task_summary[task_type] = {"total": 0, "successful": 0}
        task_summary[task_type]["total"] += 1
        if result["success"]:
            task_summary[task_type]["successful"] += 1

    logging.info(f"Overall: {successful}/{total} tasks completed successfully")
    for task_type, stats in task_summary.items():
        logging.info(f"{task_type}: {stats['successful']}/{stats['total']} successful")

    logging.info(
        f"Instance-specific result files saved under: {output_dir}/<instance_id>/"
    )
    logging.info(f"Trajectories saved in individual instance directories")


# Worker function for multiprocessing
def worker_process_instance(args):
    """
    Worker function to process a single test instance in a separate process.
    This function needs to be at module level for multiprocessing to work.
    """
    test_instance, agent_config, prompts_config, model_config, output_dir = args

    worker_logger = logging.getLogger(f"worker_{test_instance.instance_id}")
    worker_logger.setLevel(logging.INFO)

    # Create a handler that writes to both stdout and a file
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter(
            f"[Worker-{test_instance.instance_id}] %(asctime)s - %(levelname)s - %(message)s"
        )
    )
    worker_logger.addHandler(console_handler)

    # Also log to a worker-specific file
    instance_output_dir = os.path.join(output_dir, test_instance.instance_id)
    os.makedirs(instance_output_dir, exist_ok=True)
    run_log_filename = config_manager.get("output.run_log", "run.log")
    worker_log_file = os.path.join(instance_output_dir, run_log_filename)
    file_handler = logging.FileHandler(worker_log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    worker_logger.addHandler(file_handler)

    worker_logger.info(f"🔧 Worker started for instance: {test_instance.instance_id}")
    start_time = time.time()

    try:
        # Create a new pipeline instance for this worker
        pipeline = IssueSolvingPipeline(agent_config, prompts_config, model_config)

        worker_logger.info(f"📋 Processing {test_instance.task_type.value} task...")
        result = process_single_instance(test_instance, pipeline, output_dir)

        elapsed_time = time.time() - start_time
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        worker_logger.info(f"🏁 {status} - Completed in {elapsed_time:.1f}s")

        return result

    except Exception as e:
        elapsed_time = time.time() - start_time
        worker_logger.error(f"❌ FAILED after {elapsed_time:.1f}s: {e}")
        return create_error_result(
            test_instance.instance_id, test_instance.task_type, str(e)
        )


def run_pipeline():
    """Run the BugHunter pipeline with given parameters"""
    config = config_manager.get_config()

    setup_main()

    # Load test data
    try:
        test_instances = load_test_data()
        logging.info(f"Loaded {len(test_instances)} test instances")
        for instance in test_instances:
            logging.info(f"  - {instance.instance_id} ({instance.task_type.value})")
    except Exception as e:
        logging.error(f"Failed to load test data: {e}")
        sys.exit(1)

    # Initialize pipeline from config
    model_config = config["model"]
    logging.info(f"Using model: {model_config['name']}")

    # Handle environment variable substitution for api_key
    api_key = model_config["api_key"]
    if api_key.startswith("{") and api_key.endswith("}"):
        env_var_name = api_key[1:-1]  # Remove the curly braces
        api_key = os.getenv(env_var_name)
        if not api_key:
            logging.error(
                f"Environment variable {env_var_name} is required but not set"
            )
            sys.exit(1)
        logging.info(f"Using API key from environment variable: {env_var_name}")

    agent_config = AgentConfig(
        model_name=model_config["name"],
        temperature=model_config["generation"]["temperature"],
        max_tokens=model_config["generation"]["max_tokens"],
        max_iterations=config["system"]["max_iterations"],
    )
    logging.info(
        f"Agent config: max_iterations={agent_config.max_iterations}, temperature={agent_config.temperature}"
    )

    prompts_config = config.get("prompts", {})
    pipeline = IssueSolvingPipeline(agent_config, prompts_config, model_config)

    # Get user-configurable output directory from config
    base_output_dir = config.get("output", {}).get("output_dir")

    if base_output_dir is None:
        # Default to current working directory if not specified
        base_output_dir = os.getcwd()
    else:
        # Expand relative paths to absolute paths
        if not os.path.isabs(base_output_dir):
            base_output_dir = os.path.abspath(base_output_dir)

    # Create the base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    logging.info(f"Using output directory: {base_output_dir}")

    # Results file path in the base output directory
    output_path = os.path.join(base_output_dir, "results.json")

    # Use base_output_dir for instance-specific subdirectories
    output_dir = base_output_dir

    # Get number of workers from config
    num_workers = config["system"].get("num_workers", 1)

    # Solve issues - use parallel or sequential processing based on num_workers
    results = []
    start_time = time.time()

    if num_workers > 1 and len(test_instances) > 1:
        logging.info(
            f"🚀 Starting parallel processing with {num_workers} workers for {len(test_instances)} instances"
        )

        # Prepare arguments for worker processes
        worker_args = [
            (instance, agent_config, prompts_config, model_config, output_dir)
            for instance in test_instances
        ]

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_instance = {
                executor.submit(worker_process_instance, args): args[0]
                for args in worker_args
            }

            completed_count = 0
            total_count = len(test_instances)

            # Process results as they complete
            for future in as_completed(future_to_instance):
                test_instance = future_to_instance[future]
                completed_count += 1

                try:
                    result_dict = future.result()
                    results.append(result_dict)

                    status = "✅ SUCCESS" if result_dict["success"] else "❌ FAILED"
                    logging.info(
                        f"[{completed_count}/{total_count}] {status} - {test_instance.instance_id}"
                    )

                    save_intermediate_results(results, output_path)

                except Exception as e:
                    logging.error(
                        f"[{completed_count}/{total_count}] ❌ FAILED - {test_instance.instance_id}: {e}"
                    )
                    error_result = create_error_result(
                        test_instance.instance_id, test_instance.task_type, str(e)
                    )
                    results.append(error_result)

                # Log progress update
                elapsed_time = time.time() - start_time
                avg_time_per_task = elapsed_time / completed_count
                estimated_remaining = avg_time_per_task * (
                    total_count - completed_count
                )

                logging.info(
                    f"Progress: {completed_count}/{total_count} completed "
                    f"(elapsed: {elapsed_time:.1f}s, est. remaining: {estimated_remaining:.1f}s)"
                )
    else:
        # Sequential processing
        if num_workers > 1:
            logging.info("📝 Only one instance to process, using sequential processing")
        else:
            logging.info(
                f"📝 Starting sequential processing for {len(test_instances)} instances"
            )

        for i, test_instance in enumerate(test_instances, 1):
            logging.info(
                f"[{i}/{len(test_instances)}] Starting: {test_instance.instance_id}"
            )

            result_dict = process_single_instance(test_instance, pipeline, output_dir)
            results.append(result_dict)

            status = "✅ SUCCESS" if result_dict["success"] else "❌ FAILED"
            logging.info(
                f"[{i}/{len(test_instances)}] {status} - {test_instance.instance_id}"
            )

            save_intermediate_results(results, output_path)

            # Log progress for sequential processing
            elapsed_time = time.time() - start_time
            if i < len(test_instances):
                avg_time_per_task = elapsed_time / i
                estimated_remaining = avg_time_per_task * (len(test_instances) - i)
                logging.info(
                    f"Progress: {i}/{len(test_instances)} completed "
                    f"(elapsed: {elapsed_time:.1f}s, est. remaining: {estimated_remaining:.1f}s)"
                )

    # Final timing and summary
    total_time = time.time() - start_time
    logging.info(f"🏁 Pipeline completed in {total_time:.1f} seconds")

    # Print summary
    print_results_summary(results, output_dir)

    # Save fixed output files for testing
    try:
        save_fixed_output_files(results, output_dir, config)
        logging.info("📁 Fixed output files saved successfully")
    except Exception as e:
        logging.error(f"Failed to save fixed output files: {e}")


def main():
    """Main entry point for the BugHunter application with subcommands"""
    parser = create_argument_parser()
    args, unknown_args = parser.parse_known_args()

    config_path = getattr(args, "config", None)
    if config_path:
        config_manager.load_config(config_path)
    else:
        config_manager.load_config()
    handle_overrides(unknown_args)

    setup_logging()
    logging.info("Starting BugHunter pipeline")

    # If no command is provided, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "run":
            # Call the run pipeline function directly with parameters
            run_pipeline()

        elif args.command == "setup":
            # Run setup
            print(f"🚀 Running BugHunter setup with config: {args.config}")
            setup_main()

        elif args.command == "evaluate":
            if not args.eval_type:
                parser.error("evaluate command requires a type (patch or location)")

            if args.eval_type == "patch":
                print(f"📊 Evaluating patch results from: {args.results}")
                from bughunter.evaluation.patch_evaluation import run_patch_evaluation

                run_patch_evaluation(args)

            elif args.eval_type == "location":
                print(f"📊 Evaluating location results from: {args.results}")
                from bughunter.evaluation.location_evaluation import run_evaluation

                run_evaluation(args)

            else:
                parser.error(f"Unknown evaluation type: {args.eval_type}")

        else:
            parser.error(f"Unknown command: {args.command}")

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        sys.exit(1)
