"""
Configuration Utilities for BugHunter

Handles CLI argument parsing, config overrides, and configuration processing.
"""

import argparse
from typing import Dict, Any, List
from .manager import config_manager


def parse_string_args(unknown_args: List[str]) -> Dict[str, Any]:
    """Parse unknown command line arguments as config overrides"""
    config_overrides = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            key = arg[2:]  # Remove --
            if "=" in key:
                # Handle --key=value format
                key, value = key.split("=", 1)
                config_overrides[key] = value
            else:
                # Handle --key value format
                if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith(
                    "--"
                ):
                    config_overrides[key] = unknown_args[i + 1]
                    i += 1  # Skip the value
                else:
                    # Flag without value, treat as True
                    config_overrides[key] = True
        i += 1
    return config_overrides


def extract_cli_overrides(cli_args) -> Dict[str, Any]:
    """Extract config override arguments from CLI args (exclude standard arguments)"""
    if not cli_args:
        return {}

    excluded_args = {
        "config",
        "command",
        "results",
        "output",
        "detailed",
        "test_data",
        "eval_type",
    }
    return {
        k: v
        for k, v in vars(cli_args).items()
        if v is not None and k not in excluded_args
    }


def apply_cli_overrides(cli_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply CLI argument overrides to config using dot notation"""
    # Convert nested overrides to dot notation format
    dot_notation_overrides = {}

    for key, value in cli_overrides.items():
        if key == "config":  # Skip the config file argument
            continue

        # Convert string values to appropriate types
        if isinstance(value, str):
            if value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "", 1).isdigit():
                value = float(value)

        dot_notation_overrides[key] = value

    # Use the new set_config_overrides method
    if dot_notation_overrides:
        config_manager.set_config_overrides(dot_notation_overrides)
    return config_manager.get_config()


def handle_overrides(cli_args=None) -> Dict[str, Any]:
    """
    Process CLI arguments and apply configuration overrides.

    Args:
        config_file: Path to the configuration file
        cli_args: Parsed CLI arguments (optional)

    Returns:
        Final configuration dictionary with CLI overrides applied
    """
    args = parse_string_args(cli_args) if cli_args else {}
    filtered_args = extract_cli_overrides(args)
    return apply_cli_overrides(filtered_args)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the main argument parser with all subcommands"""
    parser = argparse.ArgumentParser(
        description="BugHunter - AI Agent for Bug Analysis"
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", metavar="COMMAND"
    )

    # Run subcommand
    run_parser = subparsers.add_parser(
        "run", help="Run the BugHunter pipeline on test instances"
    )
    run_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Configuration file path (default: config.yaml)",
    )

    # Setup subcommand
    setup_parser = subparsers.add_parser(
        "setup", help="Setup and validate the BugHunter environment"
    )
    setup_parser.add_argument(
        "--config",
        default="config.yaml",
        help="Configuration file path (default: config.yaml)",
    )

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate BugHunter results")
    eval_subparsers = eval_parser.add_subparsers(
        dest="eval_type", help="Evaluation type", metavar="TYPE"
    )

    # Patch evaluation
    patch_eval_parser = eval_subparsers.add_parser(
        "patch", help="Evaluate fix patch results"
    )
    patch_eval_parser.add_argument("results", help="Path to results.json file")
    patch_eval_parser.add_argument(
        "--output", help="Save detailed evaluation results to file"
    )
    patch_eval_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed results for each instance",
    )

    # Location evaluation
    location_eval_parser = eval_subparsers.add_parser(
        "location", help="Evaluate locate_bug task results"
    )
    location_eval_parser.add_argument("results", help="Path to results.json file")
    location_eval_parser.add_argument(
        "--test-data", help="Path to test data file (default: data/test_set.yaml)"
    )
    location_eval_parser.add_argument(
        "--output", help="Save detailed evaluation results to file"
    )
    location_eval_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed results for each instance",
    )

    return parser
