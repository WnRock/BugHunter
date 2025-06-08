"""
BugHunter - Global Configuration Manager

Provides centralized configuration management for the entire application.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """
    Global configuration manager using singleton pattern.
    Ensures only one instance exists and provides centralized config access.
    """

    _instance: Optional["ConfigManager"] = None
    _config: Optional[Dict[str, Any]] = None
    _config_file_path: Optional[str] = None

    def __new__(cls) -> "ConfigManager":
        """Ensure singleton pattern - only one instance exists."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def load_config(self, config_file: str = "config.yaml") -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_file: Path to config file (default: config.yaml)

        Returns:
            Dict containing configuration data

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        # Convert to absolute path if relative
        if not os.path.isabs(config_file):
            # Look for config file starting from the project root
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent  # Go up to BugHunter root
            config_file = project_root / config_file

        config_file = str(config_file)

        # Check if we need to reload (file changed or first load)
        if self._config is None or self._config_file_path != config_file:
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Configuration file not found: {config_file}")

            try:
                with open(config_file, "r") as f:
                    self._config = yaml.safe_load(f)
                self._config_file_path = config_file
                logging.debug(f"Configuration loaded from: {config_file}")
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Invalid YAML in config file {config_file}: {e}")

        return self._config

    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration. If not loaded, loads default config.yaml.

        Returns:
            Dict containing configuration data
        """
        if self._config is None:
            self.load_config()
        return self._config

    def set_config_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Set multiple configuration overrides using a dictionary.
        These values will directly overwrite the corresponding config values.

        Args:
            overrides: Dictionary with dot-notation keys and their override values
                      Example: {"system.num_workers": 5, "model.temperature": 0.7}
        """
        for key_path, value in overrides.items():
            # Directly set the value in the config
            self.set(key_path, value)

        logging.debug(f"Applied config overrides: {overrides}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation path.

        Args:
            key_path: Dot-separated path to config value (e.g., "model.name", "system.max_iterations")
            default: Default value if path not found

        Returns:
            Configuration value or default

        Examples:
            config_manager.get("model.name")
            config_manager.get("system.max_iterations", 10)
            config_manager.get("docker.pull_timeout")
        """
        config = self.get_config()

        # Split the path and traverse the config dict
        keys = key_path.split(".")
        current = config

        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation path.
        Note: This only modifies the in-memory config, doesn't save to file.

        Args:
            key_path: Dot-separated path to config value
            value: Value to set
        """
        config = self.get_config()

        keys = key_path.split(".")
        current = config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value
        current[keys[-1]] = value

    def reload_config(self, config_file: str = None) -> Dict[str, Any]:
        """
        Force reload configuration from file.

        Args:
            config_file: Optional new config file path

        Returns:
            Dict containing reloaded configuration data
        """
        if config_file:
            self._config_file_path = config_file

        # Clear cached config to force reload
        self._config = None

        return self.load_config(config_file or self._config_file_path or "config.yaml")

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance. Mainly for testing purposes."""
        cls._instance = None


# Global instance for easy access
config_manager = ConfigManager()


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Convenience function to get a specific config value.

    Args:
        key_path: Dot-separated path to config value
        default: Default value if not found

    Returns:
        Configuration value or default
    """
    return config_manager.get(key_path, default)
