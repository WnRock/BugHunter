"""
Docker container management for BugHunter
"""

import time
import docker
import logging
from bughunter.core.models import ExecutionResult


class DockerManager:
    """Manages Docker container operations"""

    def __init__(self, timeout: int = 300):
        self.client = docker.from_env()
        self.container = None
        self.timeout = timeout

    def start_container(self, image_name: str) -> bool:
        """Start a container from the given image"""
        try:
            # Remove existing container if it exists
            self.stop_container()

            # Pull the image if it doesn't exist locally
            try:
                self.client.images.get(image_name)
            except docker.errors.ImageNotFound:
                logging.info(f"Pulling image {image_name}...")
                self.client.images.pull(image_name)

            # Start the container
            self.container = self.client.containers.run(
                image_name,
                command="sleep infinity",  # Keep container running
                detach=True,
                working_dir="/",
                tty=True,
            )

            # Wait for container to be ready
            time.sleep(2)

            logging.info(f"Container {self.container.short_id} started successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to start container: {e}")
            return False

    def execute_command(self, command: str) -> ExecutionResult:
        """Execute a command in the container"""
        if not self.container:
            return ExecutionResult(False, "", "No container running", -1)

        try:
            # Execute command
            exec_result = self.container.exec_run(
                command,
                stdout=True,
                stderr=True,
                tty=False,
                workdir="/",
                environment={"TERM": "xterm-256color"},
            )

            return ExecutionResult(
                success=exec_result.exit_code == 0,
                stdout=exec_result.output.decode("utf-8", errors="replace"),
                stderr="",  # stderr is combined with stdout in exec_run
                exit_code=exec_result.exit_code,
            )

        except Exception as e:
            logging.error(f"Failed to execute command '{command}': {e}")
            return ExecutionResult(False, "", str(e), -1)

    def stop_container(self):
        """Stop and remove the container"""
        if self.container:
            try:
                self.container.stop()
                self.container.remove()
                logging.info(f"Container {self.container.short_id} stopped and removed")
            except Exception as e:
                logging.error(f"Failed to stop container: {e}")
            finally:
                self.container = None
