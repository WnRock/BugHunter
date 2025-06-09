"""
Task definitions for BugHunter
"""

import logging
from abc import ABC, abstractmethod
from bughunter.core.agent import IssueAgent
from bughunter.core.docker_manager import DockerManager
from bughunter.core.trajectory_recorder import TrajectoryRecorder
from bughunter.core.models import TestInstance, TaskResult, AgentConfig


class BaseTask(ABC):
    """Base class for all task types"""

    def __init__(
        self,
        agent_config: AgentConfig,
        prompts_config: dict = None,
        model_config: dict = None,
    ):
        self.agent_config = agent_config
        self.prompts_config = prompts_config
        self.model_config = model_config
        self.docker_manager = DockerManager()
        # Initialize trajectory recorder - will be set in _run_task_loop
        self.trajectory_recorder = None
        self.agent = None  # Will be initialized with trajectory recorder

    @abstractmethod
    def execute(self, test_instance: TestInstance) -> TaskResult:
        """Execute the task"""
        pass

    def _run_task_loop(self, test_instance: TestInstance) -> TaskResult:
        """Common task execution loop"""
        logging.info(
            f"Starting {test_instance.task_type.value} task: {test_instance.instance_id}"
        )

        # Initialize trajectory recorder with the actual docker image name
        self.trajectory_recorder = TrajectoryRecorder(
            environment=test_instance.image_name
        )

        # Initialize agent with trajectory recorder, prompts config, and model config
        self.agent = IssueAgent(
            self.agent_config,
            self.trajectory_recorder,
            self.prompts_config,
            self.model_config,
        )

        # Start container
        if not self.docker_manager.start_container(test_instance.image_name):
            return TaskResult(
                success=False,
                instance_id=test_instance.instance_id,
                task_type=test_instance.task_type,
                iterations=0,
                result_data={},
                error="Failed to start container",
            )

        try:
            # Initialize conversation
            agent_response = self.agent.initialize_conversation(test_instance)

            iteration = 0
            while iteration < self.agent_config.max_iterations:
                logging.info(f"Iteration {iteration + 1}")

                # Check if task is completed
                completion_result = self.agent.check_completion(
                    agent_response, test_instance.task_type
                )
                if completion_result:
                    logging.info(f"Task completed: {test_instance.task_type.value}")

                    # Finalize trajectory
                    self.trajectory_recorder.finalize_trajectory(
                        exit_status="submitted",
                        submission=self._extract_submission_from_result(
                            completion_result
                        ),
                    )

                    result = self._create_success_result(
                        test_instance, completion_result, iteration + 1
                    )
                    result.trajectory = self.trajectory_recorder.trajectory
                    return result

                # Extract command from agent response
                command = self.agent.extract_command(agent_response)
                if not command:
                    logging.warning("No command found in agent response")
                    break

                logging.info(f"Executing command: {command}")

                # Execute command
                result = self.docker_manager.execute_command(command)

                # Send result back to agent
                agent_response = self.agent.send_command_result(command, result)

                iteration += 1

            # Finalize trajectory for incomplete tasks
            self.trajectory_recorder.finalize_trajectory(
                exit_status="max_iterations_reached", submission=""
            )

            result = TaskResult(
                success=False,
                instance_id=test_instance.instance_id,
                task_type=test_instance.task_type,
                iterations=iteration,
                result_data={},
                error=f"Max iterations ({self.agent_config.max_iterations}) reached without solution",
            )
            result.trajectory = self.trajectory_recorder.trajectory
            return result

        finally:
            # Always clean up container
            self.docker_manager.stop_container()

    def _extract_submission_from_result(self, completion_result: str) -> str:
        """Extract the final submission/patch from completion result"""
        # Look for patch content or diff after PATCH_READY
        if "PATCH_READY" in completion_result:
            lines = completion_result.split("\n")
            patch_lines = []
            found_patch = False

            for line in lines:
                if "PATCH_READY" in line:
                    found_patch = True
                    continue
                if found_patch:
                    patch_lines.append(line)

            return "\n".join(patch_lines).strip()

        # For location tasks, return the location info
        elif "LOCATION_FOUND" in completion_result:
            return completion_result

        return completion_result

    @abstractmethod
    def _create_success_result(
        self, test_instance: TestInstance, completion_result: str, iterations: int
    ) -> TaskResult:
        """Create a success result with task-specific data"""
        pass
