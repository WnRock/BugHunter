"""
LLM Agent management for the Agent-based Issue Solving System
"""

import os
from openai import OpenAI
from typing import Optional
from bughunter.config.manager import config_manager
from bughunter.core.trajectory_recorder import TrajectoryRecorder
from bughunter.core.models import ExecutionResult, AgentConfig, TaskType, TestInstance


class IssueAgent:
    """Handles communication with LLM agent for issue solving"""

    def __init__(
        self,
        config: AgentConfig,
        trajectory_recorder: Optional[TrajectoryRecorder] = None,
        prompts_config: Optional[dict] = None,
        model_config: Optional[dict] = None,
    ):
        self.config = config
        self.model_config = model_config or self._load_default_model_config()

        # Handle environment variable substitution for api_key
        api_key = self.model_config.get("api_key", os.getenv("OPENAI_API_KEY"))
        if api_key and api_key.startswith("{") and api_key.endswith("}"):
            env_var_name = api_key[1:-1]  # Remove the curly braces
            api_key = os.getenv(env_var_name)

        # Set up OpenAI client with custom base URL if provided
        client_kwargs = {"api_key": api_key}
        if "base_url" in self.model_config:
            client_kwargs["base_url"] = self.model_config["base_url"]

        self.client = OpenAI(**client_kwargs)
        self.conversation_history = []
        self.trajectory_recorder = trajectory_recorder
        self.prompts_config = prompts_config or self._load_default_prompts_config()

    def _load_default_prompts_config(self) -> dict:
        """Load prompts configuration using global config manager"""
        try:
            return config_manager.get(
                "prompts",
                {
                    "directory": "bughunter/prompts",
                    "fix_bug": "fix_bug.txt",
                    "locate_bug": "locate_bug.txt",
                    "fix_with_location": "fix_with_location.txt",
                },
            )
        except Exception:
            # Fallback to default configuration
            return {
                "directory": "bughunter/prompts",
                "fix_bug": "fix_bug.txt",
                "locate_bug": "locate_bug.txt",
                "fix_with_location": "fix_with_location.txt",
            }

    def _load_default_model_config(self) -> dict:
        """Load model configuration using global config manager"""
        try:
            return config_manager.get(
                "model",
                {
                    "name": "DeepSeek-V3",
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key": "{OPENAI_API_KEY}",
                },
            )
        except Exception:
            # Fallback to default configuration
            return {
                "name": "DeepSeek-V3",
                "base_url": "https://api.deepseek.com/v1",
                "api_key": "{OPENAI_API_KEY}",
            }

    def initialize_conversation(
        self,
        test_instance: TestInstance,
    ) -> str:
        """Initialize the conversation with the problem statement and task type"""

        # Load task-specific system prompt (without instance-specific info)
        system_prompt = self._load_system_prompt(test_instance.task_type)

        # Create instance-specific user message
        instance_message = self._create_instance_message(test_instance)

        # Initialize conversation with system prompt and instance info
        self.conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_message},
        ]

        # Record system prompt and instance info in trajectory
        if self.trajectory_recorder:
            self.trajectory_recorder.record_system_prompt(system_prompt)
            self.trajectory_recorder.record_user_message(
                instance_message, "instance_info"
            )

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=self.conversation_history,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        assistant_message = response.choices[0].message.content

        # DON'T add full assistant response to conversation history - save for trajectory only
        # Extract command and add only that to conversation history if present
        command = self.extract_command(assistant_message)
        if command:
            self.conversation_history.append({"role": "assistant", "content": f"```bash\n{command}\n```"})

        # Record full response in trajectory
        if self.trajectory_recorder:
            self.trajectory_recorder.record_api_call(
                tokens_sent=response.usage.prompt_tokens if response.usage else 0,
                tokens_received=(
                    response.usage.completion_tokens if response.usage else 0
                ),
            )

            self.trajectory_recorder.record_agent_response(
                response=assistant_message, action=command if command else ""
            )

        return assistant_message

    def send_command_result(self, command: str, result: ExecutionResult) -> str:
        """Send command execution result to the agent and get next instruction"""
        # Create brief command result message for conversation history
        brief_message = f"Command:\n```bash\n{command}\n```\nResult: Exit code {result.exit_code}"
        if result.stdout.strip():
            brief_message += f"\nStdout: {result.stdout}"
        if result.stderr.strip():
            brief_message += f"\nStderr: {result.stderr}"

        # Add only the brief command result to conversation history
        self.conversation_history.append({"role": "user", "content": brief_message})

        # Record full command execution details in trajectory
        if self.trajectory_recorder:
            self.trajectory_recorder.record_command_execution(command, result)

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=self.conversation_history,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        assistant_message = response.choices[0].message.content

        # DON'T add full assistant response to conversation history - only extract command
        next_command = self.extract_command(assistant_message)
        if next_command:
            self.conversation_history.append({"role": "assistant", "content": f"```bash\n{next_command}\n```"})

        # Record full response in trajectory
        if self.trajectory_recorder:
            self.trajectory_recorder.record_api_call(
                tokens_sent=response.usage.prompt_tokens if response.usage else 0,
                tokens_received=(
                    response.usage.completion_tokens if response.usage else 0
                ),
            )

            self.trajectory_recorder.record_agent_response(
                response=assistant_message, action=next_command if next_command else ""
            )

        return assistant_message

    def _load_system_prompt(self, task_type: TaskType) -> str:
        """Load the system prompt template for the task type (without instance-specific info)"""

        # Determine the prompt file name based on task type
        prompt_file_map = {
            TaskType.FIX_BUG: self.prompts_config.get("fix_bug", "fix_bug.txt"),
            TaskType.LOCATE_BUG: self.prompts_config.get(
                "locate_bug", "locate_bug.txt"
            ),
            TaskType.FIX_WITH_LOCATION: self.prompts_config.get(
                "fix_with_location", "fix_with_location.txt"
            ),
        }

        if task_type not in prompt_file_map:
            raise ValueError(f"Unknown task type: {task_type}")

        # Build the full path to the prompt file
        prompt_dir = self.prompts_config.get("directory", "bughunter/prompts")
        prompt_file = prompt_file_map[task_type]

        # Handle relative paths by making them relative to the project root
        if not os.path.isabs(prompt_dir):
            project_root = os.path.join(os.path.dirname(__file__), "..", "..")
            prompt_path = os.path.join(project_root, prompt_dir, prompt_file)
        else:
            prompt_path = os.path.join(prompt_dir, prompt_file)

        # Load the prompt template from file
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        except Exception as e:
            raise Exception(f"Error reading prompt file {prompt_path}: {e}")

        return system_prompt

    def _create_instance_message(self, test_instance: TestInstance) -> str:
        """Create instance-specific user message with problem statement and hints"""

        message = f"Instance ID: {test_instance.instance_id}\n\n"
        message += f"Problem Statement:\n{test_instance.problem_statement}"

        # Add location hint if available and task type is FIX_WITH_LOCATION
        if (
            test_instance.gold_target_file
            and test_instance.task_type == TaskType.FIX_WITH_LOCATION
        ):
            if isinstance(test_instance.gold_target_file, list):
                # Multiple files
                files_text = ", ".join(test_instance.gold_target_file)
                message += (
                    f"\n\nLocation Hint: The bug is likely in these files: {files_text}"
                )
            else:
                # Single file
                message += f"\n\nLocation Hint: The bug is likely in this file: {test_instance.gold_target_file}"

        return message

    def extract_command(self, agent_response: str) -> Optional[str]:
        """Extract bash command from agent response"""
        lines = agent_response.split("\n")

        # Look for bash code blocks
        in_code_block = False
        command_lines = []

        for line in lines:
            if line.strip().startswith("```bash") or line.strip().startswith("```sh"):
                in_code_block = True
                continue
            elif line.strip() == "```" and in_code_block:
                break
            elif in_code_block:
                command_lines.append(line)

        if command_lines:
            return "\n".join(command_lines).strip()

        # Fallback: look for lines starting with $ or containing common commands
        for line in lines:
            line = line.strip()
            if line.startswith("$ "):
                return line[2:]
            elif any(
                line.startswith(cmd)
                for cmd in [
                    "ls",
                    "cat",
                    "grep",
                    "find",
                    "cd",
                    "pwd",
                    "git",
                    "make",
                    "npm",
                    "python",
                ]
            ):
                return line

        return None

    def check_completion(
        self, agent_response: str, task_type: TaskType
    ) -> Optional[str]:
        """Check if the task is completed and extract the result"""
        if task_type == TaskType.FIX_BUG or task_type == TaskType.FIX_WITH_LOCATION:
            if "PATCH_READY" in agent_response:
                return agent_response
        elif task_type == TaskType.LOCATE_BUG:
            if "LOCATION_CANDIDATES" in agent_response:
                return agent_response

        return None
