"""
BugHunter - Main Pipeline
"""

import os
import sys
import yaml
import json
import logging
import argparse
from typing import List, Dict, Any
from bughunter.evaluation import run_evaluation
from bughunter.tasks.fix_bug_task import FixBugTask
from bughunter.utils.setup import main as setup_main
from bughunter.tasks.locate_bug_task import LocateBugTask
from concurrent.futures import ProcessPoolExecutor, as_completed
from bughunter.tasks.fix_with_location_task import FixWithLocationTask
from bughunter.core.models import TestInstance, TaskResult, AgentConfig, TaskType
from bughunter.utils.output_manager import (
    save_fixed_output_files,
    save_instance_result_files,
    save_trajectory_file,
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


def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not os.path.isabs(config_file):
        config_file = os.path.join(os.path.dirname(__file__), "..", config_file)

    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def load_test_data(config: Dict[str, Any]) -> List[TestInstance]:
    """Load test instances from YAML file specified in config"""
    yaml_file = config["tasks"]["test_data_file"]

    # Adjust path to be relative to project root
    if not os.path.isabs(yaml_file):
        yaml_file = os.path.join(os.path.dirname(__file__), "..", yaml_file)

    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    instances = []

    # Use task type from config with validation
    try:
        task_type = TaskType(config["tasks"]["task_type"])
    except ValueError as e:
        raise ValueError(
            f"Invalid task_type in config: {config['tasks']['task_type']}. Must be one of: {[t.value for t in TaskType]}"
        )

    # Optional instance filtering from config
    target_instances = config["tasks"].get("target_instances", None)

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


def setup_logging(config: Dict[str, Any]):
    """Setup logging configuration from config"""
    log_level = config["system"]["log_level"]

    # Note: Removed automatic logs directory creation - logs will be handled per test case
    # Just use console logging for the main pipeline
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


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


def run_setup_if_needed(config_file_path):
    """Run setup automatically before main execution"""
    try:
        print("🔧 Running automatic setup check...")
        setup_main(config_file_path)
        print("✅ Setup check completed successfully\n")
    except SystemExit as e:
        if e.code != 0:
            print("❌ Setup failed. Please run 'python main.py setup' first.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Setup error: {e}")
        print("Please run 'python main.py setup' first.")
        sys.exit(1)


# Worker function for multiprocessing
def worker_process_instance(args):
    """
    Worker function to process a single test instance in a separate process.
    This function needs to be at module level for multiprocessing to work.
    """
    test_instance, agent_config, prompts_config, model_config, output_dir = args

    # Set up logging for this worker process
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Worker-{test_instance.instance_id}] %(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    try:
        # Create a new pipeline instance for this worker
        pipeline = IssueSolvingPipeline(agent_config, prompts_config, model_config)

        logging.info(
            f"Processing {test_instance.task_type.value} task: {test_instance.instance_id}"
        )
        result = pipeline.solve_issue(test_instance)

        # Save instance-specific result files (for batch functionality)
        save_instance_result_files(result, output_dir)

        # Save trajectory to separate file (always enabled by default)
        save_trajectory_file(result, output_dir)

        return convert_result_to_dict(result)
    except Exception as e:
        logging.error(f"Error processing instance {test_instance.instance_id}: {e}")
        # Return a failed result
        return {
            "success": False,
            "instance_id": test_instance.instance_id,
            "task_type": test_instance.task_type.value,
            "iterations": 0,
            "result_data": {},
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Agent-based Issue Solving System")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # Setup subcommand
    setup_parser = subparsers.add_parser(
        "setup", help="Run initial setup and validation"
    )
    setup_parser.add_argument("--config", help="Path to config file (optional)")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the bug solving pipeline")
    run_parser.add_argument(
        "--config", default="config.yaml", help="Path to config file"
    )

    # Evaluate subcommand
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate locate_bug results against gold truth"
    )
    eval_parser.add_argument(
        "results", help="Path to results.json file with LLM outputs"
    )
    eval_parser.add_argument(
        "--test-data", help="Path to test_set.yaml file (default: data/test_set.yaml)"
    )
    eval_parser.add_argument(
        "--output", help="Path to save detailed evaluation results (JSON format)"
    )
    eval_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed results for each instance",
    )

    args = parser.parse_args()

    # Handle setup command
    if args.command == "setup":
        project_root = os.path.join(os.path.dirname(__file__), "..")
        original_cwd = os.getcwd()
        os.chdir(project_root)
        try:
            setup_main(args.config)
        finally:
            os.chdir(original_cwd)
        return

    # Handle evaluate command
    if args.command == "evaluate":
        # Setup basic logging for evaluation
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        run_evaluation(args)
        return

    # Handle run command - automatically run setup first
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Run setup check before main execution
    project_root = os.path.join(os.path.dirname(__file__), "..")
    original_cwd = os.getcwd()
    os.chdir(project_root)
    try:
        run_setup_if_needed(args.config)
    finally:
        os.chdir(original_cwd)

    setup_logging(config)

    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    # Load test data
    try:
        test_instances = load_test_data(config)
        logging.info(f"Loaded {len(test_instances)} test instances")
    except Exception as e:
        logging.error(f"Failed to load test data: {e}")
        sys.exit(1)

    # Initialize pipeline from config
    model_config = config["model"]

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

    agent_config = AgentConfig(
        model_name=model_config["name"],
        temperature=model_config["generation"]["temperature"],
        max_tokens=model_config["generation"]["max_tokens"],
        max_iterations=config["system"]["max_iterations"],
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

    if num_workers > 1 and len(test_instances) > 1:
        logging.info(
            f"Using parallel processing with {num_workers} workers for {len(test_instances)} instances"
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

            # Process results as they complete
            for future in as_completed(future_to_instance):
                test_instance = future_to_instance[future]
                try:
                    result_dict = future.result()

                    # Include only trajectory summary in main results for performance
                    if "trajectory" in result_dict:
                        trajectory_summary = {
                            "environment": result_dict["trajectory"]["environment"],
                            "steps_count": len(result_dict["trajectory"]["trajectory"]),
                            "history_count": len(result_dict["trajectory"]["history"]),
                            "exit_status": (
                                result_dict["trajectory"]["info"]["exit_status"]
                                if result_dict["trajectory"]["info"]
                                else "unknown"
                            ),
                        }
                        result_dict["trajectory_summary"] = trajectory_summary
                        del result_dict["trajectory"]

                    results.append(result_dict)

                    # Save intermediate results
                    with open(output_path, "w") as f:
                        json.dump(results, f, indent=2)

                except Exception as e:
                    logging.error(
                        f"Error processing instance {test_instance.instance_id}: {e}"
                    )
                    # Add failed result
                    results.append(
                        {
                            "success": False,
                            "instance_id": test_instance.instance_id,
                            "task_type": test_instance.task_type.value,
                            "iterations": 0,
                            "result_data": {},
                            "error": str(e),
                        }
                    )
    else:
        # Sequential processing (original logic)
        if num_workers > 1:
            logging.info("Only one instance to process, using sequential processing")
        else:
            logging.info(
                f"Using sequential processing for {len(test_instances)} instances"
            )

        for test_instance in test_instances:
            logging.info(
                f"Processing {test_instance.task_type.value} task: {test_instance.instance_id}"
            )
            result = pipeline.solve_issue(test_instance)

            # Save instance-specific result files (for batch functionality)
            save_instance_result_files(result, output_dir)

            # Save trajectory to separate file (always enabled by default)
            save_trajectory_file(result, output_dir)

            # Convert result for JSON output
            result_dict = convert_result_to_dict(result)
            # Include only trajectory summary in main results for performance
            if "trajectory" in result_dict:
                trajectory_summary = {
                    "environment": result_dict["trajectory"]["environment"],
                    "steps_count": len(result_dict["trajectory"]["trajectory"]),
                    "history_count": len(result_dict["trajectory"]["history"]),
                    "exit_status": (
                        result_dict["trajectory"]["info"]["exit_status"]
                        if result_dict["trajectory"]["info"]
                        else "unknown"
                    ),
                }
                result_dict["trajectory_summary"] = trajectory_summary
                del result_dict["trajectory"]

            results.append(result_dict)

            # Save intermediate results
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    # Print summary
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

    # Save fixed output files for testing
    try:
        save_fixed_output_files(results, output_dir, config)
    except Exception as e:
        logging.error(f"Failed to save fixed output files: {e}")
