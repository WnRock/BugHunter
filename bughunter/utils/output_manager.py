"""
Output Manager - Handles all file saving operations for BugHunter
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from bughunter.core.models import TaskResult


def save_fixed_output_files(results: List[Dict[str, Any]], output_dir: str, config: Dict[str, Any]):
    """Save output files with fixed names for automated testing"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save run log with fixed name
    save_run_log(results, output_dir, config)
    
    # 2. Save trajectory with fixed name
    save_fixed_trajectory(results, output_dir)
    
    # 3. Save unified result file (paths for location tasks, patches for fix tasks)
    save_llm_result(results, output_dir)


def save_run_log(results: List[Dict[str, Any]], output_dir: str, config: Dict[str, Any]):
    """Save a summary log of the run with fixed filename"""
    log_path = os.path.join(output_dir, "run.log")
    
    with open(log_path, "w") as f:
        f.write(f"BugHunter Pipeline Run Log\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"{'='*50}\n\n")
        
        # Overall statistics
        successful = len([r for r in results if r["success"]])
        total = len(results)
        f.write(f"Overall Results: {successful}/{total} tasks completed successfully\n\n")
        
        # Task type breakdown
        task_summary = {}
        for result in results:
            task_type = result["task_type"]
            if task_type not in task_summary:
                task_summary[task_type] = {"total": 0, "successful": 0}
            task_summary[task_type]["total"] += 1
            if result["success"]:
                task_summary[task_type]["successful"] += 1
        
        f.write("Task Type Breakdown:\n")
        for task_type, stats in task_summary.items():
            f.write(f"  {task_type}: {stats['successful']}/{stats['total']} successful\n")
        f.write("\n")
        
        # Individual task results
        f.write("Individual Task Results:\n")
        f.write("-" * 30 + "\n")
        for result in results:
            f.write(f"Instance ID: {result['instance_id']}\n")
            f.write(f"Task Type: {result['task_type']}\n")
            f.write(f"Success: {result['success']}\n")
            f.write(f"Iterations: {result['iterations']}\n")
            if result.get('error'):
                f.write(f"Error: {result['error']}\n")
            f.write(f"Result Data: {json.dumps(result['result_data'], indent=2)}\n")
            f.write("-" * 30 + "\n")
    
    logging.info(f"Run log saved to {log_path}")


def save_fixed_trajectory(results: List[Dict[str, Any]], output_dir: str):
    """Save trajectory data with fixed filename"""
    trajectory_path = os.path.join(output_dir, "trajectory.json")
    
    # Collect all trajectories or trajectory summaries
    trajectories_data = {
        "run_timestamp": datetime.now().isoformat(),
        "total_tasks": len(results),
        "trajectories": []
    }
    
    for result in results:
        trajectory_info = {
            "instance_id": result["instance_id"],
            "task_type": result["task_type"],
            "success": result["success"]
        }
        
        # Include full trajectory if available, otherwise summary
        if "trajectory" in result:
            trajectory_info["trajectory"] = result["trajectory"]
        elif "trajectory_summary" in result:
            trajectory_info["trajectory_summary"] = result["trajectory_summary"]
        
        trajectories_data["trajectories"].append(trajectory_info)
    
    with open(trajectory_path, "w") as f:
        json.dump(trajectories_data, f, indent=2)
    
    logging.info(f"Trajectory data saved to {trajectory_path}")


def save_llm_result(results: List[Dict[str, Any]], output_dir: str):
    """Save LLM results with task-specific content: paths for location tasks, patches for fix tasks"""
    result_path = os.path.join(output_dir, "result.txt")
    
    with open(result_path, "w") as f:
        for result in results:
            result_data = result.get('result_data', {})
            
            if result['task_type'] == 'locate_bug':
                # For locate tasks, output the path/location
                if 'location' in result_data:
                    f.write(str(result_data['location']))
                elif 'file_path' in result_data:
                    location = str(result_data['file_path'])
                    if 'line_number' in result_data:
                        location += f":{result_data['line_number']}"
                    f.write(location)
                else:
                    # Fallback for other location formats
                    f.write(json.dumps(result_data, indent=2))
                    
            elif result['task_type'] in ['fix_bug', 'fix_with_location']:
                # For fix tasks, output the patch
                if 'patch' in result_data:
                    f.write(result_data['patch'])
                elif 'solution' in result_data:
                    f.write(str(result_data['solution']))
                else:
                    # Fallback for other fix formats
                    f.write(json.dumps(result_data, indent=2))
            
            # Only add newline if there are more results
            if result != results[-1]:
                f.write("\n")
    
    logging.info(f"LLM results saved to {result_path}")


def save_instance_result_files(result: TaskResult, base_output_dir: str):
    """Save all result files for a single instance in its own directory"""
    instance_dir = os.path.join(base_output_dir, result.instance_id)
    os.makedirs(instance_dir, exist_ok=True)
    
    # Save instance-specific log
    log_path = os.path.join(instance_dir, "run.log")
    with open(log_path, "w") as f:
        f.write(f"BugHunter Instance Run Log\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Instance ID: {result.instance_id}\n")
        f.write(f"Task Type: {result.task_type.value}\n")
        f.write(f"{'='*50}\n\n")
        
        f.write(f"Success: {result.success}\n")
        f.write(f"Iterations: {result.iterations}\n")
        if result.error:
            f.write(f"Error: {result.error}\n")
        f.write(f"\nResult Data:\n{json.dumps(result.result_data, indent=2)}\n")
    
    # Save instance-specific result file with unified content
    result_path = os.path.join(instance_dir, "result.txt")
    with open(result_path, "w") as f:
        result_data = result.result_data
        
        if result.task_type.value == 'locate_bug':
            # For locate tasks, output just the path/location
            if 'location' in result_data:
                f.write(str(result_data['location']))
            elif 'file_path' in result_data:
                location = str(result_data['file_path'])
                if 'line_number' in result_data:
                    location += f":{result_data['line_number']}"
                f.write(location)
            else:
                # Fallback for other location formats
                f.write(json.dumps(result_data, indent=2))
                
        elif result.task_type.value in ['fix_bug', 'fix_with_location']:
            # For fix tasks, output just the patch
            if 'patch' in result_data:
                f.write(result_data['patch'])
            elif 'solution' in result_data:
                f.write(str(result_data['solution']))
            elif 'fix' in result_data:
                f.write(str(result_data['fix']))
            else:
                # Fallback for other fix formats
                f.write(json.dumps(result_data, indent=2))
    
    logging.info(f"Instance result files saved to {instance_dir}")


def save_trajectory_file(result: TaskResult, base_output_dir: str):
    """Save trajectory to a separate file"""
    if not result.trajectory:
        return

    # Create instance-specific directory
    instance_dir = os.path.join(base_output_dir, result.instance_id)
    os.makedirs(instance_dir, exist_ok=True)

    # Save trajectory file in instance directory
    trajectory_filename = f"{result.task_type.value}_trajectory.json"
    trajectory_path = os.path.join(instance_dir, trajectory_filename)

    with open(trajectory_path, "w") as f:
        json.dump(result.trajectory.to_dict(), f, indent=2)

    logging.info(f"Trajectory saved to {trajectory_path}")