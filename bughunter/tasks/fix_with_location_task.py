"""
Fix Bug with Location Task - Fixes bugs when location information is provided
"""

import re
from bughunter.tasks.base_task import BaseTask
from bughunter.core.models import TestInstance, TaskResult, TaskType


class FixWithLocationTask(BaseTask):
    """Task for fixing bugs when location hint is provided"""

    def __init__(self, agent_config, prompts_config=None, model_config=None):
        super().__init__(agent_config, prompts_config, model_config)

    def execute(self, test_instance: TestInstance) -> TaskResult:
        """Execute the fix with location task"""
        test_instance.task_type = TaskType.FIX_WITH_LOCATION
        return self._run_task_loop(test_instance)

    def _create_success_result(
        self, test_instance: TestInstance, completion_result: str, iterations: int
    ) -> TaskResult:
        """Create success result with patch data and location context"""
        # Extract patch from completion result
        patch_content = self._extract_patch(completion_result)
        fix_summary = self._extract_fix_summary(completion_result)

        return TaskResult(
            success=True,
            instance_id=test_instance.instance_id,
            task_type=test_instance.task_type,
            iterations=iterations,
            result_data={
                "patch": patch_content,
                "fix": fix_summary,
                "gold_target_file": test_instance.gold_target_file,
                "full_response": completion_result,
            },
        )

    def _extract_patch(self, completion_result: str) -> str:
        """Extract the actual patch content from the completion result"""
        # Look for content after PATCH_READY
        patch_start = completion_result.find("PATCH_READY")
        if patch_start != -1:
            return completion_result[patch_start + len("PATCH_READY") :].strip()

        # Look for diff format patches
        diff_patterns = [
            r"diff --git.*?(?=diff --git|\Z)",
            r"--- .*?\n\+\+\+ .*?\n@@.*?@@.*?(?=diff|---|\Z)",
            r"@@.*?@@.*?(?=@@|\Z)",
        ]

        for pattern in diff_patterns:
            matches = re.findall(pattern, completion_result, re.DOTALL)
            if matches:
                return "\n".join(matches).strip()

        # Look for code blocks that might contain patches
        code_blocks = re.findall(
            r"```(?:diff|patch)?\n(.*?)\n```", completion_result, re.DOTALL
        )
        if code_blocks:
            return "\n".join(code_blocks).strip()

        return completion_result.strip()

    def _extract_fix_summary(self, completion_result: str) -> str:
        """Extract a summary of the targeted fix"""
        # Look for explanation before the patch
        lines = completion_result.split("\n")
        fix_lines = []

        for line in lines:
            line = line.strip()
            if any(
                keyword in line.lower()
                for keyword in ["patch_ready", "diff --git", "@@", "+++"]
            ):
                break
            if line and not line.startswith("#") and len(line) > 10:
                fix_lines.append(line)
                if len(fix_lines) >= 4:  # Limit summary length
                    break

        return " ".join(fix_lines) if fix_lines else "Targeted bug fix applied"
