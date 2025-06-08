"""
Locate Bug Task - Identifies the specific location of the bug
"""

import re
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
        # Extract location from completion result
        location_data = self._extract_location(completion_result)
        analysis_summary = self._extract_analysis_summary(completion_result)

        return TaskResult(
            success=True,
            instance_id=test_instance.instance_id,
            task_type=test_instance.task_type,
            iterations=iterations,
            result_data={
                "location": location_data,
                "analysis": analysis_summary,
                "full_response": completion_result
            },
        )

    def _extract_location(self, completion_result: str) -> dict:
        """Extract the bug location from the completion result"""
        # Look for LOCATION_FOUND pattern
        location_pattern = r"LOCATION_FOUND:\s*([^:\s]+):?(\d+)?"
        match = re.search(location_pattern, completion_result)

        if match:
            file_path = match.group(1)
            line_number = int(match.group(2)) if match.group(2) else None

            return {
                "file_path": file_path,
                "line_number": line_number,
                "raw_location": match.group(0),
                "confidence": "high"
            }

        # Look for file:line patterns
        file_line_patterns = [
            r"([/\w\.-]+\.[a-zA-Z]+):(\d+)",
            r"line (\d+) (?:in|of) ([/\w\.-]+\.[a-zA-Z]+)",
            r"([/\w\.-]+\.[a-zA-Z]+) at line (\d+)"
        ]
        
        for pattern in file_line_patterns:
            matches = re.findall(pattern, completion_result)
            if matches:
                if len(matches[0]) == 2:
                    file_path, line_number = matches[0]
                    if pattern == file_line_patterns[1]:  # line X in file format
                        line_number, file_path = file_path, line_number
                    
                    return {
                        "file_path": file_path,
                        "line_number": int(line_number),
                        "raw_location": f"{file_path}:{line_number}",
                        "confidence": "medium"
                    }

        # Fallback: extract any file paths mentioned
        file_patterns = [
            r"(/[\w/\.-]+\.[a-zA-Z]+)",
            r"([\w\.-]+\.[a-zA-Z]+)"
        ]
        
        for pattern in file_patterns:
            files = re.findall(pattern, completion_result)
            if files:
                return {
                    "file_path": files[0],
                    "line_number": None,
                    "raw_location": files[0],
                    "confidence": "low"
                }

        return {
            "file_path": None,
            "line_number": None,
            "raw_location": completion_result[:200] + "..." if len(completion_result) > 200 else completion_result,
            "confidence": "unknown"
        }

    def _extract_analysis_summary(self, completion_result: str) -> str:
        """Extract a summary of the bug analysis"""
        # Look for explanation before LOCATION_FOUND
        lines = completion_result.split('\n')
        analysis_lines = []
        
        for line in lines:
            line = line.strip()
            if 'LOCATION_FOUND' in line:
                break
            if line and not line.startswith('#') and len(line) > 15:
                analysis_lines.append(line)
                if len(analysis_lines) >= 3:  # Limit analysis length
                    break
        
        return ' '.join(analysis_lines) if analysis_lines else "Bug location analysis completed"
