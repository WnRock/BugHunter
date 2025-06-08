"""
Setup utility for the Agent-based Issue Solving System
Handles initial setup and validation of the environment.
"""

import os
import sys
import yaml
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from bughunter.config.manager import config_manager


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True


def check_docker():
    """Check if Docker is installed and accessible"""
    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"✅ Docker is available: {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker command failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Docker is not installed or not in PATH")
        return False


def check_docker_permissions():
    """Check if user can run Docker without sudo"""
    try:
        result = subprocess.run(
            ["docker", "ps"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("✅ Docker permissions are correct")
            return True
        else:
            print("❌ Docker permission denied. Run: sudo usermod -aG docker $USER")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Docker daemon may not be running")
        return False


def install_dependencies():
    """Install Python dependencies"""
    try:
        print("📦 Installing Python dependencies...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def setup_env_file():
    """Setup environment file from template"""
    env_file = Path(".env")
    env_example = Path(".env.example")

    if env_file.exists():
        print("✅ .env file already exists")
        return True

    if env_example.exists():
        # Copy template
        with open(env_example) as f:
            content = f.read()

        with open(env_file, "w") as f:
            f.write(content)

        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your API keys")
        return True
    else:
        print("❌ .env.example template not found")
        return False


def validate_config():
    """Validate configuration file"""
    try:
        config = config_manager.get_config()

        # Basic validation
        required_sections = ["system", "model", "docker", "output", "tasks"]
        for section in required_sections:
            if section not in config:
                print(
                    f"❌ Missing section '{section}' in {config_manager._config_file_path}"
                )
                return False

        # Validate system settings
        system = config.get("system", {})
        if (
            not isinstance(system.get("max_iterations"), int)
            or system["max_iterations"] <= 0
        ):
            print("❌ system.max_iterations must be a positive integer")
            return False

        if system.get("log_level") not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            print("❌ system.log_level must be one of: DEBUG, INFO, WARNING, ERROR")
            return False

        # Validate model settings (changed from models to model)
        model = config.get("model", {})
        if not model.get("name"):
            print("❌ model.name is required")
            return False

        # Validate docker settings
        docker = config.get("docker", {})
        if (
            not isinstance(docker.get("pull_timeout"), int)
            or docker["pull_timeout"] <= 0
        ):
            print("❌ docker.pull_timeout must be a positive integer")
            return False

        # Output section is optional now, no required fields

        # Validate tasks settings
        tasks = config.get("tasks", {})
        valid_task_types = ["fix_bug", "locate_bug", "fix_with_location"]
        if tasks.get("task_type") not in valid_task_types:
            print(f"❌ tasks.task_type must be one of: {', '.join(valid_task_types)}")
            return False

        # Validate prompts settings
        prompts = config.get("prompts", {})
        if not prompts.get("directory"):
            print("❌ prompts.directory is required")
            return False

        prompt_files = ["fix_bug", "locate_bug", "fix_with_location"]
        for prompt_type in prompt_files:
            if not prompts.get(prompt_type):
                print(f"❌ prompts.{prompt_type} is required")
                return False

        print(f"✅ {config_manager._config_file_path} is valid")
        return True
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML in {config_manager._config_file_path}: {e}")
        return False


def validate_test_data(config_file_path=None):
    """Validate test data file"""
    try:
        # Load config to get test data file path
        config_file = (
            Path(config_file_path) if config_file_path else Path("config.yaml")
        )

        if not config_file.exists():
            print(f"❌ {config_file} not found")
            return False

        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Get test data file path from config
        test_data_file = config.get("tasks", {}).get("test_data_file")

        if not test_data_file:
            print("❌ tasks.test_data_file not specified in config")
            return False

        test_file = Path(test_data_file)

    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML in config file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading config file: {e}")
        return False

    if not test_file.exists():
        print(f"❌ {test_file} not found")
        return False

    try:
        with open(test_file) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, list):
            print(f"❌ {test_file} should contain a list")
            return False

        required_fields = ["image_name", "instance_id", "problem_statement"]
        for i, item in enumerate(data):
            for field in required_fields:
                if field not in item:
                    print(f"❌ Missing field '{field}' in test item {i}")
                    return False

        print(f"✅ {test_file} is valid ({len(data)} test instances)")
        return True
    except yaml.YAMLError as e:
        print(f"❌ Invalid YAML in {test_file}: {e}")
        return False


def test_docker_image():
    """Test Docker functionality with a simple image"""
    try:
        print("🐳 Testing Docker with hello-world image...")
        result = subprocess.run(
            ["docker", "run", "--rm", "hello-world"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print("✅ Docker test successful")
            return True
        else:
            print(f"❌ Docker test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Docker test timed out")
        return False
    except Exception as e:
        print(f"❌ Docker test error: {e}")
        return False


def validate_environment_variables():
    """Validate required environment variables"""
    # Load environment variables
    load_dotenv()

    required_vars = {"OPENAI_API_KEY": "OpenAI API key for LLM access"}

    missing_vars = []
    invalid_vars = []

    for var_name, description in required_vars.items():
        value = os.getenv(var_name)
        if not value:
            missing_vars.append(f"{var_name} ({description})")
        elif var_name == "OPENAI_API_KEY":
            # Basic validation for OpenAI API key format
            if not value.startswith(("sk-", "sk-proj-")) or len(value) < 20:
                invalid_vars.append(f"{var_name} (invalid format)")

    if missing_vars:
        print(f"❌ Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        return False

    if invalid_vars:
        print(f"❌ Invalid environment variables:")
        for var in invalid_vars:
            print(f"   - {var}")
        return False

    print("✅ All required environment variables are set and valid")
    return True


def validate_prompt_files():
    """Validate that prompt files exist and are readable"""
    try:
        prompts = config_manager.get("prompts")
        prompt_dir = config_manager.get("prompts.directory")

        if not Path(prompt_dir).exists():
            print(f"❌ Prompts directory not found: {prompt_dir}")
            return False

        # Check each prompt file
        prompt_files = {
            "fix_bug": prompts.get("fix_bug", "fix_bug.txt"),
            "locate_bug": prompts.get("locate_bug", "locate_bug.txt"),
            "fix_with_location": prompts.get(
                "fix_with_location", "fix_with_location.txt"
            ),
        }

        missing_files = []
        for prompt_type, filename in prompt_files.items():
            file_path = Path(prompt_dir) / filename
            if not file_path.exists():
                missing_files.append(f"{prompt_type}: {file_path}")
            elif not file_path.is_file():
                missing_files.append(f"{prompt_type}: {file_path} (not a file)")
            else:
                # Try to read the file to ensure it's readable
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if not content:
                            missing_files.append(
                                f"{prompt_type}: {file_path} (empty file)"
                            )
                except Exception as e:
                    missing_files.append(
                        f"{prompt_type}: {file_path} (read error: {e})"
                    )

        if missing_files:
            print("❌ Missing or invalid prompt files:")
            for file_info in missing_files:
                print(f"   - {file_info}")
            return False

        print(f"✅ All prompt files are present and readable in {prompt_dir}")
        return True

    except Exception as e:
        print(f"❌ Error validating prompt files: {e}")
        return False


def create_output_directories():
    """Create necessary output directories based on configuration"""
    try:
        base_output_dir = config_manager.get("output.output_dir")

        if base_output_dir is None:
            # Default to current working directory if not specified
            base_output_dir = os.getcwd()
            print(f"   - Using default output directory: {base_output_dir}")
        else:
            # Expand relative paths to absolute paths
            if not os.path.isabs(base_output_dir):
                base_output_dir = os.path.abspath(base_output_dir)
            print(f"   - Using configured output directory: {base_output_dir}")

        # Create the base output directory
        os.makedirs(base_output_dir, exist_ok=True)

        # Note: Logs directory creation removed - logs will be created per test case
        print("   - Logs will be created under individual test case directories")

        print("✅ All output directories created successfully")
        return True

    except Exception as e:
        print(f"❌ Error creating output directories: {e}")
        return False


def main():
    """Main setup function"""
    print("🚀 Setting up Agent-based Issue Solving System\n")

    checks = [
        ("Python Version", check_python_version),
        ("Docker Installation", check_docker),
        ("Docker Permissions", check_docker_permissions),
        ("Environment File", setup_env_file),
        ("Configuration", lambda: validate_config()),
        ("Test Data", validate_test_data),
        ("Dependencies", install_dependencies),
        ("Docker Test", test_docker_image),
        ("Environment Variables", validate_environment_variables),
        ("Prompt Files", lambda: validate_prompt_files()),
        ("Output Directories", lambda: create_output_directories()),
    ]

    failed_checks = []

    for name, check_func in checks:
        print(f"\n📋 {name}:")
        if not check_func():
            failed_checks.append(name)

    print("\n" + "=" * 50)
    if failed_checks:
        print(f"❌ Setup incomplete. Failed checks: {', '.join(failed_checks)}")
        print("\nPlease fix the issues above and run setup again.")
        sys.exit(1)
    else:
        print("✅ Setup completed successfully!")


if __name__ == "__main__":
    main()
