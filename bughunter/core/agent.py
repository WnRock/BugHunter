"""
LLM Agent management for the Agent-based Issue Solving System
"""

import os
import yaml
from openai import OpenAI
from typing import Optional
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
        """Load prompts configuration from config.yaml"""
        try:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "config.yaml"
            )
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config.get(
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
        """Load model configuration from config.yaml"""
        try:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "config.yaml"
            )
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config.get(
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

        # Load task-specific prompt
        task_prompt = self._load_task_prompt(test_instance)

        self.conversation_history = [{"role": "system", "content": task_prompt}]

        # Record system prompt in trajectory
        if self.trajectory_recorder:
            self.trajectory_recorder.record_system_prompt(task_prompt)

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=self.conversation_history,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        assistant_message = response.choices[0].message.content
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_message}
        )

        # Record API call and agent response
        if self.trajectory_recorder:
            self.trajectory_recorder.record_api_call(
                tokens_sent=response.usage.prompt_tokens if response.usage else 0,
                tokens_received=(
                    response.usage.completion_tokens if response.usage else 0
                ),
            )

            # Extract command for trajectory recording
            command = self.extract_command(assistant_message)
            self.trajectory_recorder.record_agent_response(
                response=assistant_message, action=command if command else ""
            )

        return assistant_message

    def send_command_result(self, command: str, result: ExecutionResult) -> str:
        """Send command execution result to the agent and get next instruction"""
        user_message = f"""Command executed: {command}
Exit code: {result.exit_code}
STDOUT:
{result.stdout}
STDERR:
{result.stderr}

Please provide the next command or your analysis."""

        self.conversation_history.append({"role": "user", "content": user_message})

        # Record command execution in trajectory
        if self.trajectory_recorder:
            self.trajectory_recorder.record_command_execution(command, result)

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=self.conversation_history,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        assistant_message = response.choices[0].message.content
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_message}
        )

        # Record API call and agent response
        if self.trajectory_recorder:
            self.trajectory_recorder.record_api_call(
                tokens_sent=response.usage.prompt_tokens if response.usage else 0,
                tokens_received=(
                    response.usage.completion_tokens if response.usage else 0
                ),
            )

            # Extract command for trajectory recording
            next_command = self.extract_command(assistant_message)
            self.trajectory_recorder.record_agent_response(
                response=assistant_message, action=next_command if next_command else ""
            )

        return assistant_message

    def _load_task_prompt(
        self,
        test_instance: TestInstance,
    ) -> str:
        """Load the appropriate prompt template for the task type"""

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

        if test_instance.task_type not in prompt_file_map:
            raise ValueError(f"Unknown task type: {test_instance.task_type}")

        # Build the full path to the prompt file
        prompt_dir = self.prompts_config.get("directory", "bughunter/prompts")
        prompt_file = prompt_file_map[test_instance.task_type]

        # Handle relative paths by making them relative to the project root
        if not os.path.isabs(prompt_dir):
            project_root = os.path.join(os.path.dirname(__file__), "..", "..")
            prompt_path = os.path.join(project_root, prompt_dir, prompt_file)
        else:
            prompt_path = os.path.join(prompt_dir, prompt_file)

        # Load the prompt template from file
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_template = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        except Exception as e:
            raise Exception(f"Error reading prompt file {prompt_path}: {e}")

        # Prepare location hint from gold_target_file
        location_hint_text = ""
        if test_instance.gold_target_file:
            if isinstance(test_instance.gold_target_file, list):
                # Multiple files
                files_text = ", ".join(test_instance.gold_target_file)
                location_hint_text = (
                    f"\nLocation Hint: The bug is likely in these files: {files_text}"
                )
            else:
                # Single file
                location_hint_text = f"\nLocation Hint: The bug is likely in this file: {test_instance.gold_target_file}"

        # Format the template with the provided variables
        try:
            formatted_prompt = prompt_template.format(
                instance_id=test_instance.instance_id,
                problem_statement=test_instance.problem_statement,
                location_hint=location_hint_text,
            )
        except KeyError as e:
            raise ValueError(
                f"Missing placeholder in prompt template {prompt_path}: {e}"
            )

        return formatted_prompt

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
            if "LOCATION_FOUND" in agent_response:
                return agent_response

        return None
