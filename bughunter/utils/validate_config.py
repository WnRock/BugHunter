"""
Configuration validator for the Agent-based Issue Solving System
Validates configuration files and environment setup.
"""

import os
import sys
import yaml
from pathlib import Path


class ConfigValidator:
    """Validates system configuration and environment"""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate_env_file(self) -> bool:
        """Validate .env file"""
        env_path = Path(".env")

        if not env_path.exists():
            self.errors.append(".env file not found")
            return False

        required_vars = ["OPENAI_API_KEY"]
        optional_vars = ["LOG_LEVEL", "MAX_ITERATIONS", "DEFAULT_MODEL"]

        env_vars = {}
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        env_vars[key.strip()] = value.strip()
        except Exception as e:
            self.errors.append(f"Error reading .env file: {e}")
            return False

        # Check required variables
        for var in required_vars:
            if var not in env_vars or not env_vars[var]:
                self.errors.append(f"Required environment variable {var} not set")
            elif env_vars[var] == "your_openai_api_key_here":
                self.warnings.append(
                    f"Environment variable {var} still has placeholder value"
                )

        # Check optional variables
        for var in optional_vars:
            if var not in env_vars:
                self.warnings.append(f"Optional environment variable {var} not set")

        return len([e for e in self.errors if ".env" in e]) == 0

    def validate_config_yaml(self) -> bool:
        """Validate config.yaml structure and values"""
        config_path = Path("config.yaml")

        if not config_path.exists():
            self.errors.append("config.yaml not found")
            return False

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML in config.yaml: {e}")
            return False

        # Required sections and their required fields
        required_structure = {
            "system": ["max_iterations", "timeout_seconds", "log_level"],
            "model": ["name", "base_url", "api_key", "generation"],
            "docker": ["pull_timeout", "execution_timeout"],
            "output": [],  # Output section exists but no required fields since we use defaults
        }

        for section, fields in required_structure.items():
            if section not in config:
                self.errors.append(f"Missing section '{section}' in config.yaml")
                continue

            for field in fields:
                if field not in config[section]:
                    self.errors.append(
                        f"Missing field '{field}' in section '{section}'"
                    )

        # Validate model section structure
        if "model" in config:
            model = config["model"]

            # Check generation subsection
            if "generation" in model:
                generation = model["generation"]
                required_generation_fields = ["temperature", "max_tokens"]

                for field in required_generation_fields:
                    if field not in generation:
                        self.errors.append(
                            f"Missing field '{field}' in model.generation section"
                        )

                # Validate temperature
                if "temperature" in generation:
                    temp = generation["temperature"]
                    if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                        self.errors.append(
                            "model.generation.temperature must be a number between 0 and 2"
                        )

                # Validate max_tokens
                if "max_tokens" in generation:
                    max_tokens = generation["max_tokens"]
                    if not isinstance(max_tokens, int) or max_tokens <= 0:
                        self.errors.append(
                            "model.generation.max_tokens must be a positive integer"
                        )

            # Validate api_key format (should support environment variable references)
            if "api_key" in model:
                api_key = model["api_key"]
                if not api_key or not isinstance(api_key, str):
                    self.errors.append("model.api_key must be a non-empty string")
                elif not (api_key.startswith("{") and api_key.endswith("}")) and not api_key.startswith("sk-"):
                    self.warnings.append("model.api_key should either reference an environment variable with {VAR_NAME} or be a valid API key")

        # Validate specific values
        if "system" in config:
            system = config["system"]

            if "max_iterations" in system:
                if (
                    not isinstance(system["max_iterations"], int)
                    or system["max_iterations"] <= 0
                ):
                    self.errors.append("max_iterations must be a positive integer")

            if "log_level" in system:
                valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
                if system["log_level"] not in valid_levels:
                    self.errors.append(f"log_level must be one of {valid_levels}")

        return len([e for e in self.errors if "config.yaml" in e]) == 0

    def validate_test_data(self) -> bool:
        """Validate test_set.yaml structure"""
        test_path = Path("test_set.yaml")

        if not test_path.exists():
            self.errors.append("test_set.yaml not found")
            return False

        try:
            with open(test_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML in test_set.yaml: {e}")
            return False

        if not isinstance(data, list):
            self.errors.append("test_set.yaml must contain a list of test instances")
            return False

        if len(data) == 0:
            self.warnings.append("test_set.yaml is empty")

        required_fields = ["image_name", "instance_id", "problem_statement"]

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                self.errors.append(f"Test item {i} must be a dictionary")
                continue

            for field in required_fields:
                if field not in item:
                    self.errors.append(
                        f"Test item {i} missing required field '{field}'"
                    )
                elif not isinstance(item[field], str) or not item[field].strip():
                    self.errors.append(
                        f"Test item {i} field '{field}' must be a non-empty string"
                    )

            # Validate image name format
            if "image_name" in item:
                image_name = item["image_name"]
                if ":" not in image_name:
                    self.warnings.append(
                        f"Test item {i}: image_name '{image_name}' has no tag"
                    )

        return len([e for e in self.errors if "test_set.yaml" in e]) == 0

    def validate_directories(self) -> bool:
        """Check if required directories exist or can be created"""
        required_dirs = ["conversations", "patches", "logs"]

        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.warnings.append(f"Created missing directory: {dir_name}")
                except Exception as e:
                    self.errors.append(f"Cannot create directory {dir_name}: {e}")

        return len([e for e in self.errors if "directory" in e.lower()]) == 0

    def validate_permissions(self) -> bool:
        """Check file permissions"""
        files_to_check = ["main.py", "utils/setup.py"]

        for file_path in files_to_check:
            path = Path(file_path)
            if path.exists() and not os.access(path, os.X_OK):
                self.warnings.append(f"File {file_path} is not executable")

        return True

    def run_validation(self) -> bool:
        """Run all validations"""
        print("🔍 Validating configuration...\n")

        validations = [
            ("Environment file", self.validate_env_file),
            ("Configuration YAML", self.validate_config_yaml),
            ("Test data", self.validate_test_data),
            ("Directories", self.validate_directories),
            ("Permissions", self.validate_permissions),
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
        sys.exit(0)
    else:
        validator.print_summary()
        print("\nPlease fix the errors above before running the system.")
        sys.exit(1)


if __name__ == "__main__":
    main()
