"""
Locate Bug Task - Identifies the specific location of the bug
"""

import re
from typing import List, Dict, Any
from bughunter.tasks.base_task import BaseTask
from bughunter.core.models import TestInstance, TaskResult, TaskType


class LocateBugTask(BaseTask):
    """Task for locating bugs and identifying their positions"""

    def __init__(self, agent_config, prompts_config=None, model_config=None):
        super().__init__(agent_config, prompts_config, model_config)

    def execute(self, test_instance: TestInstance) -> TaskResult:
        """Execute the locate bug task"""
        test_instance.task_type = TaskType.LOCATE_BUG
        return self._run_task_loop(test_instance)

    def _create_success_result(
        self, test_instance: TestInstance, completion_result: str, iterations: int
    ) -> TaskResult:
        """Create success result with location data"""
        # Extract multiple locations from completion result
        location_candidates = self._extract_location_candidates(completion_result)
        analysis_summary = self._extract_analysis_summary(completion_result)

        return TaskResult(
            success=True,
            instance_id=test_instance.instance_id,
            task_type=test_instance.task_type,
            iterations=iterations,
            result_data={
                "location_candidates": location_candidates,
                "primary_location": (
                    location_candidates[0] if location_candidates else None
                ),
                "analysis": analysis_summary,
                "full_response": completion_result,
            },
        )

    def _extract_location_candidates(
        self, completion_result: str
    ) -> List[Dict[str, Any]]:
        """Extract multiple location candidates from the completion result"""
        candidates = []

        # Look for LOCATION_CANDIDATES section
        candidates_pattern = r"LOCATION_CANDIDATES:\s*\n(.*?)(?:\n\n|\Z)"
        match = re.search(
            candidates_pattern, completion_result, re.DOTALL | re.IGNORECASE
        )

        if match:
            candidates_text = match.group(1)
            # Parse numbered list of candidates
            candidate_lines = re.findall(
                r"(\d+)\.\s*([^-\n]+?)(?:\s*-\s*(.+?))?(?=\n\d+\.|\Z)",
                candidates_text,
                re.DOTALL,
            )

            for rank, location_str, explanation in candidate_lines:
                location_data = self._parse_location_string(location_str.strip())
                if location_data["file_path"]:
                    location_data.update(
                        {
                            "rank": int(rank),
                            "explanation": explanation.strip() if explanation else "",
                            "confidence": self._determine_confidence(int(rank)),
                        }
                    )
                    candidates.append(location_data)

        return candidates

    def _parse_location_string(self, location_str: str) -> Dict[str, Any]:
        """Parse a location string to extract file path and line number"""
        # Try file:line format
        file_line_match = re.match(r"([^:]+):(\d+)", location_str)
        if file_line_match:
            return {
                "file_path": file_line_match.group(1).strip(),
                "line_number": int(file_line_match.group(2)),
                "raw_location": location_str,
            }

        # Try just file path
        file_match = re.match(r"([^:]+)", location_str)
        if file_match:
            return {
                "file_path": file_match.group(1).strip(),
                "line_number": None,
                "raw_location": location_str,
            }

        return {"file_path": None, "line_number": None, "raw_location": location_str}

    def _determine_confidence(self, rank: int) -> str:
        """Determine confidence level based on rank"""
        if rank == 1:
            return "high"
        elif rank <= 2:
            return "medium"
        else:
            return "low"

    def _extract_analysis_summary(self, completion_result: str) -> str:
        """Extract a summary of the bug analysis"""
        # Look for explanation before LOCATION_CANDIDATES
        lines = completion_result.split("\n")
        analysis_lines = []

        for line in lines:
            line = line.strip()
            if "LOCATION_CANDIDATES" in line:
                break
            if line and not line.startswith("#") and len(line) > 15:
                analysis_lines.append(line)
                if len(analysis_lines) >= 3:  # Limit analysis length
                    break

        return (
            " ".join(analysis_lines)
            if analysis_lines
            else "Bug location analysis completed"
        )
