"""
Configuration validator for the Agent-based Issue Solving System
Validates configuration files and environment setup.
"""

from pathlib import Path
from .manager import config_manager


class ConfigValidator:
    """Validates system configuration and environment"""

    # Required configuration keys (dot notation)
    REQUIRED_KEYS = [
        "model.name",
        "model.base_url",
        "model.api_key",
        "model.generation.temperature",
        "model.generation.max_tokens",
        "system.max_iterations",
        "system.log_level",
        "docker.pull_timeout",
        "docker.execution_timeout",
    ]

    # Required files/directories
    REQUIRED_FILES = ["config.yaml", "data/test_set.yaml"]

    REQUIRED_DIRS = ["bughunter/prompts", "data"]

    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate_config_keys(self) -> bool:
        """Validate required configuration keys"""
        if not config_manager._config_file_path.exists():
            self.errors.append("config file not found")
            return False

        # Check required keys
        for key_path in self.REQUIRED_KEYS:
            value = config_manager.get(key_path)
            if value is None:
                self.errors.append(f"Missing required config key: {key_path}")
            elif isinstance(value, str) and not value.strip():
                self.errors.append(f"Empty value for required config key: {key_path}")

        # Check API key
        api_key = config_manager.get("model.api_key")
        if api_key and api_key == "{OPENAI_API_KEY}":
            self.warnings.append(
                "API key still has placeholder value - check your .env file"
            )

        return len([e for e in self.errors if "config" in e.lower()]) == 0

    def validate_files_and_dirs(self) -> bool:
        """Check required files and directories exist"""
        # Check required files
        for file_path in self.REQUIRED_FILES:
            if not Path(file_path).exists():
                self.errors.append(f"Required file not found: {file_path}")

        # Check required directories
        for dir_path in self.REQUIRED_DIRS:
            if not Path(dir_path).exists():
                self.errors.append(f"Required directory not found: {dir_path}")

        # Create output directory if it doesn't exist
        output_dir = Path("output")
        if not output_dir.exists():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                self.warnings.append("Created output directory")
            except Exception as e:
                self.errors.append(f"Cannot create output directory: {e}")

        return (
            len(
                [
                    e
                    for e in self.errors
                    if any(x in e.lower() for x in ["file", "directory"])
                ]
            )
            == 0
        )

    def validate_env_file(self) -> bool:
        """Check .env file exists and has OPENAI_API_KEY"""
        env_path = Path(".env")

        if not env_path.exists():
            self.warnings.append(
                ".env file not found - API key must be set in config or environment"
            )
            return True

        try:
            with open(env_path) as f:
                content = f.read()
                if "OPENAI_API_KEY=" not in content:
                    self.warnings.append("OPENAI_API_KEY not found in .env file")
                elif "OPENAI_API_KEY=your_openai_api_key_here" in content:
                    self.warnings.append(
                        "OPENAI_API_KEY has placeholder value in .env file"
                    )
        except Exception as e:
            self.warnings.append(f"Cannot read .env file: {e}")

        return True

    def run_validation(self) -> bool:
        """Run all validations"""
        print("🔍 Validating configuration...\n")

        validations = [
            ("Configuration keys", self.validate_config_keys),
            ("Files and directories", self.validate_files_and_dirs),
            ("Environment file", self.validate_env_file),
        ]

        all_valid = True

        for name, validator in validations:
            print(f"📋 {name}:")
            if validator():
                print("✅ Valid")
            else:
                print("❌ Issues found")
                all_valid = False
            print()

        return all_valid

    def print_summary(self):
        """Print validation summary"""
        print("=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)

        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All validations passed!")
        elif not self.errors:
            print(f"\n✅ Validation passed with {len(self.warnings)} warnings")
        else:
            print(
                f"\n❌ Validation failed with {len(self.errors)} errors and {len(self.warnings)} warnings"
            )


def main():
    """Main validation function"""
    validator = ConfigValidator()

    if validator.run_validation():
        validator.print_summary()
        return True
    else:
        validator.print_summary()
        print("\nPlease fix the errors above before running the system.")
        return False
