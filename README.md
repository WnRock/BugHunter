# BugHunter

An automated pipeline that uses LLM agents to solve different types of software issues in Docker containers. The system supports multiple task types including bug fixing, bug location, and targeted fixes with location hints. Now includes comprehensive trajectory recording for detailed analysis of agent interactions.

## Features

- 🤖 **Multiple Task Types**: Fix bugs, locate bugs, or fix with location hints
- 🔧 **Modular Architecture**: Clean separation of concerns with task-specific implementations
- 🐳 **Docker Integration**: Automatic container management and command execution
- 📝 **Structured Logging**: Comprehensive logging with configurable levels
- 🎯 **Selective Execution**: Run specific test instances or task types
- 📊 **Result Tracking**: JSON output with task-specific results
- 🛤️ **Trajectory Recording**: Detailed recording of agent interactions, commands, and responses
- ⚙️ **Easy Setup**: Automated setup and validation with built-in CLI command

## Task Types

### 1. Fix Bug (`fix_bug`)
**Purpose**: Analyze the problem and provide a complete patch to fix the issue.
**Output**: Complete patch or code changes
**Usage**: Default task type, good for end-to-end bug fixing

### 2. Locate Bug (`locate_bug`) 
**Purpose**: Identify the specific file and line number where the bug is located.
**Output**: File path and line number of the bug
**Usage**: When you need to find where the bug is without fixing it

### 3. Fix with Location (`fix_with_location`)
**Purpose**: Fix a bug when you already know approximately where it is located.
**Output**: Targeted patch for the specific location
**Usage**: When you have hints about bug location for more efficient fixing

## Quick Start

### Prerequisites

- Python 3.8+, tested with 3.12
- Docker
- OpenAI API key (or other supported LLM provider)

### Installation

1. Clone the repository and navigate to the project directory

2. Run the automated setup:
   ```bash
   python main.py setup
   ```
   
   This will:
   - Check Python and Docker installation
   - Validate Docker permissions
   - Install Python dependencies
   - Create environment file from template
   - Validate configuration files
   - Test Docker functionality

3. Configure your API key:
   ```bash
   # Edit .env with your API keys
   nano .env
   ```

### Usage

```bash
# Run with custom config file
python main.py run --config custom_config.yaml
```

## CLI Commands

### Setup Command
```bash
# Setup with custom config file
python main.py setup --config custom_config.yaml
```

Runs comprehensive system setup and validation:
- ✅ Checks Python version compatibility (3.8+)
- ✅ Verifies Docker installation and permissions
- ✅ Installs Python dependencies from requirements.txt
- ✅ Creates .env file from template if needed
- ✅ Validates configuration files (uses config.yaml by default, or custom file if specified)
- ✅ Tests Docker functionality with hello-world image

### Run Command
```bash
python main.py run [options]
```

Executes the bug-solving pipeline with various options for task types, models, and output configuration.

### Evaluate Command
```bash
python main.py evaluate <results_file> [options]
```

Evaluates the correctness of locate_bug results by comparing LLM outputs with gold truth data.

## Trajectory Recording

The system now records comprehensive trajectories of all agent interactions, including:

- **System prompts** and initial task setup
- **Agent responses** with extracted thoughts and actions
- **Command executions** with full output and error information
- **API call statistics** including token usage and costs
- **State tracking** of working directory and open files

### Trajectory Format

Each trajectory includes:
```json
{
    "environment": "swe_main",
    "trajectory": [
        {
            "action": "ls -F\n",
            "observation": "AUTHORS.rst\nCHANGELOG.rst\n...",
            "response": "Let's list out some of the files...",
            "state": "{\"open_file\": \"n/a\", \"working_dir\": \"/path\"}",
            "thought": "Let's explore the repository structure..."
        }
    ],
    "history": [
        {
            "message_type": "system_prompt",
            "role": "system", 
            "content": "You are an expert software engineer...",
            "agent": "primary"
        }
    ],
    "info": {
        "exit_status": "submitted",
        "submission": "diff --git a/file.py...",
        "model_stats": {
            "total_cost": 0.0,
            "instance_cost": 0.0,
            "tokens_sent": 1500,
            "tokens_received": 800,
            "api_calls": 5
        }
    }
}
```
