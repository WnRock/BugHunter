"""
BugHunter - Evaluation utilities (main module)
"""

# Import functions from separated modules
from .location_evaluation import (
    evaluate_locate_bug_results,
    run_evaluation
)
from .patch_evaluation import (
    evaluate_patch_results,
    run_patch_evaluation
)

# Re-export functions for backward compatibility
__all__ = [
    'evaluate_locate_bug_results',
    'evaluate_patch_results', 
    'run_evaluation',
    'run_patch_evaluation'
]
