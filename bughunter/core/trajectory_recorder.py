"""
Trajectory recording functionality for the Agent-based Issue Solving System
"""

import json
import os
import logging
from typing import Dict, Any
from bughunter.core.models import (
    Trajectory,
    TrajectoryStep,
    HistoryMessage,
    ModelStats,
    TrajectoryInfo,
    ExecutionResult,
)


class TrajectoryRecorder:
    """Records trajectories of agent interactions during task execution"""

    def __init__(self, environment: str = None, working_dir: str = "/"):
        if environment is None:
            raise ValueError("Environment (docker image name) must be provided")
        self.trajectory = Trajectory(environment=environment)
        self.working_dir = working_dir
        self.open_file = "n/a"
        self.model_stats = ModelStats()

    def record_system_prompt(self, content: str):
        """Record the initial system prompt"""
        self.trajectory.history.append(
            HistoryMessage(
                message_type="system_prompt",
                role="system",
                content=content,
                agent="primary",
            )
        )

    def record_agent_response(self, response: str, thought: str = "", action: str = ""):
        """Record agent response with optional thought and action"""
        # Add to history
        history_msg = HistoryMessage(
            message_type="action", role="assistant", content=response, agent="primary"
        )

        if thought:
            history_msg.thought = thought
        if action:
            history_msg.action = action

        self.trajectory.history.append(history_msg)

        # If there's an action, also create a trajectory step
        if action:
            # Extract the thought from the response or use provided thought
            extracted_thought = (
                self._extract_thought_from_response(response)
                if not thought
                else thought
            )

            step = TrajectoryStep(
                action=action,
                observation="",  # Will be filled when we get the command result
                response=response,
                state=self._get_current_state(),
                thought=extracted_thought,
            )
            self.trajectory.trajectory.append(step)

    def record_command_execution(self, command: str, result: ExecutionResult):
        """Record the execution of a command and its result"""
        # Update the last trajectory step with the observation
        if self.trajectory.trajectory:
            last_step = self.trajectory.trajectory[-1]
            if last_step.action.strip() == command.strip():
                # Format the observation
                observation = result.stdout
                if result.stderr:
                    observation += f"\nSTDERR:\n{result.stderr}"

                last_step.observation = observation

        # Add observation to history
        observation_content = f"Command executed: {command}\nExit code: {result.exit_code}\nSTDOUT:\n{result.stdout}"
        if result.stderr:
            observation_content += f"\nSTDERR:\n{result.stderr}"

        self.trajectory.history.append(
            HistoryMessage(
                message_type="observation",
                role="user",
                content=observation_content,
                agent="primary",
            )
        )

        # Update working directory if it's a cd command
        if command.strip().startswith("cd "):
            self._update_working_dir(command, result)

    def record_api_call(
        self, tokens_sent: int = 0, tokens_received: int = 0, cost: float = 0.0
    ):
        """Record API call statistics"""
        self.model_stats.api_calls += 1
        self.model_stats.tokens_sent += tokens_sent
        self.model_stats.tokens_received += tokens_received
        self.model_stats.instance_cost += cost
        self.model_stats.total_cost += cost

    def finalize_trajectory(self, exit_status: str = "submitted", submission: str = ""):
        """Finalize the trajectory with completion information"""
        # Extract additional metadata from submission for better result formatting
        submission_metadata = self._extract_submission_metadata(submission)

        self.trajectory.info = TrajectoryInfo(
            exit_status=exit_status, submission=submission, model_stats=self.model_stats
        )

        # Add submission metadata to trajectory info if available
        if hasattr(self.trajectory.info, "submission_metadata"):
            self.trajectory.info.submission_metadata = submission_metadata

    def save_trajectory(self, filepath: str):
        """Save trajectory to a JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.trajectory.to_dict(), f, indent=2)

        logging.info(f"Trajectory saved to {filepath}")

    def get_trajectory_dict(self) -> Dict[str, Any]:
        """Get trajectory as dictionary"""
        return self.trajectory.to_dict()

    def _extract_thought_from_response(self, response: str) -> str:
        """Extract thought/reasoning from agent response"""
        # Look for common patterns that indicate thinking/reasoning
        lines = response.split("\n")
        thought_lines = []

        for line in lines:
            line = line.strip()
            # Skip empty lines and code blocks
            if not line or line.startswith("```"):
                continue
            # Stop at command execution
            if any(
                cmd in line.lower()
                for cmd in ["ls", "cat", "grep", "find", "cd", "pwd"]
            ):
                break
            thought_lines.append(line)

        return (
            " ".join(thought_lines[:3])
            if thought_lines
            else "Analyzing the situation..."
        )

    def _get_current_state(self) -> str:
        """Get current state as JSON string"""
        state = {"open_file": self.open_file, "working_dir": self.working_dir}
        return json.dumps(state)

    def _update_working_dir(self, command: str, result: ExecutionResult):
        """Update working directory based on cd command result"""
        if result.success and command.strip().startswith("cd "):
            # Try to extract new directory from command
            parts = command.strip().split()
            if len(parts) > 1:
                new_dir = parts[1]
                if new_dir.startswith("/"):
                    self.working_dir = new_dir
                elif new_dir == "..":
                    self.working_dir = os.path.dirname(self.working_dir) or "/"
                else:
                    self.working_dir = os.path.join(self.working_dir, new_dir)

    def update_open_file(self, filepath: str):
        """Update the currently open file"""
        self.open_file = filepath

    def update_working_dir(self, working_dir: str):
        """Update the working directory"""
        self.working_dir = working_dir

    def _extract_submission_metadata(self, submission: str) -> Dict[str, Any]:
        """Extract metadata from submission for better result formatting"""
        metadata = {
            "type": "unknown",
            "has_patch": False,
            "has_location": False,
            "file_count": 0,
            "line_count": len(submission.split("\n")) if submission else 0,
        }

        if not submission:
            return metadata

        # Check for patch indicators
        patch_indicators = ["diff --git", "@@", "+++", "---", "PATCH_READY"]
        if any(indicator in submission for indicator in patch_indicators):
            metadata["has_patch"] = True
            metadata["type"] = "patch"

        # Check for location indicators
        location_indicators = ["LOCATION_FOUND", ".py:", ".js:", ".c:", ".cpp:"]
        if any(indicator in submission for indicator in location_indicators):
            metadata["has_location"] = True
            if metadata["type"] == "unknown":
                metadata["type"] = "location"

        # Count potential file references
        import re

        file_patterns = [
            r"[/\w\.-]+\.[a-zA-Z]+",  # file.ext pattern
            r"diff --git a/([^\s]+)",  # git diff files
        ]

        files = set()
        for pattern in file_patterns:
            files.update(re.findall(pattern, submission))

        metadata["file_count"] = len(files)

        return metadata
