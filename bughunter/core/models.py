"""
Core data models for BugHunter
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


class TaskType(Enum):
    """Types of tasks the system can perform"""

    FIX_BUG = "fix_bug"
    LOCATE_BUG = "locate_bug"
    FIX_WITH_LOCATION = "fix_with_location"


@dataclass
class TestInstance:
    """Represents a single test instance from the YAML file"""

    image_name: str
    instance_id: str
    problem_statement: str
    task_type: TaskType = TaskType.FIX_BUG
    location_hint: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of executing a command in the container"""

    success: bool
    stdout: str
    stderr: str
    exit_code: int


@dataclass
class TaskResult:
    """Result of a completed task"""

    success: bool
    instance_id: str
    task_type: TaskType
    iterations: int
    result_data: Dict[str, Any]
    error: Optional[str] = None
    trajectory: Optional["Trajectory"] = None


@dataclass
class AgentConfig:
    """Configuration for the LLM agent"""

    model_name: str = "DeepSeek-V3"
    temperature: float = 0.1
    max_tokens: int = 4000
    max_iterations: int = 50


@dataclass
class TrajectoryStep:
    """Single step in the trajectory"""

    action: str
    observation: str
    response: str
    state: str
    thought: str


@dataclass
class HistoryMessage:
    """Single message in the conversation history"""

    message_type: str  # "system_prompt", "observation", "action"
    role: str  # "system", "user", "assistant"
    content: str
    agent: str = "primary"
    thought: Optional[str] = None
    action: Optional[str] = None


@dataclass
class ModelStats:
    """Statistics about model usage"""

    total_cost: float = 0.0
    instance_cost: float = 0.0
    tokens_sent: int = 0
    tokens_received: int = 0
    api_calls: int = 0


@dataclass
class TrajectoryInfo:
    """Information about the trajectory execution"""

    exit_status: str
    submission: str
    model_stats: ModelStats


@dataclass
class Trajectory:
    """Complete trajectory of task execution"""

    environment: str
    trajectory: List[TrajectoryStep] = field(default_factory=list)
    history: List[HistoryMessage] = field(default_factory=list)
    info: Optional[TrajectoryInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary format"""
        return {
            "environment": self.environment,
            "trajectory": [
                {
                    "action": step.action,
                    "observation": step.observation,
                    "response": step.response,
                    "state": step.state,
                    "thought": step.thought,
                }
                for step in self.trajectory
            ],
            "history": [
                {
                    "message_type": msg.message_type,
                    "role": msg.role,
                    "content": msg.content,
                    "agent": msg.agent,
                    **({"thought": msg.thought} if msg.thought else {}),
                    **({"action": msg.action} if msg.action else {}),
                }
                for msg in self.history
            ],
            "info": (
                {
                    "exit_status": self.info.exit_status,
                    "submission": self.info.submission,
                    "model_stats": {
                        "total_cost": self.info.model_stats.total_cost,
                        "instance_cost": self.info.model_stats.instance_cost,
                        "tokens_sent": self.info.model_stats.tokens_sent,
                        "tokens_received": self.info.model_stats.tokens_received,
                        "api_calls": self.info.model_stats.api_calls,
                    },
                }
                if self.info
                else None
            ),
        }
