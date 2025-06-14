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
            patch_content = completion_result[patch_start + len("PATCH_READY"):].strip()
            
            # Extract diff block if it's wrapped in code blocks
            diff_block_match = re.search(r'```(?:diff)?\n(.*?)\n```', patch_content, re.DOTALL)
            if diff_block_match:
                return diff_block_match.group(1).strip()
            
            # If no code block, return content after PATCH_READY
            return patch_content

        # Look for diff format patches in code blocks
        code_blocks = re.findall(r'```(?:diff|patch)?\n(.*?)\n```', completion_result, re.DOTALL)
        for block in code_blocks:
            if '---' in block and '+++' in block and '@@' in block:
                return block.strip()

        # Look for diff format patches without code blocks
        diff_patterns = [
            r'--- .*?\n\+\+\+ .*?\n@@.*?@@.*?(?=\n---|$)',
            r'diff --git.*?(?=diff --git|$)'
        ]

        for pattern in diff_patterns:
            matches = re.findall(pattern, completion_result, re.DOTALL)
            if matches:
                return '\n'.join(matches).strip()

        # Fallback: return the full completion if no specific pattern found
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
