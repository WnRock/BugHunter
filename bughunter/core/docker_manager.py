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
        self.current_image = None  # Track the current image for restart

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
                working_dir="/home",
                tty=True,
            )

            # Store the image name for potential restart
            self.current_image = image_name

            # Wait for container to be ready
            time.sleep(2)

            logging.info(f"Container {self.container.short_id} started successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to start container: {e}")
            return False

    def _is_container_running(self) -> bool:
        """Check if the container is still running"""
        if not self.container:
            return False

        try:
            # Refresh container status
            self.container.reload()
            return self.container.status == "running"
        except docker.errors.NotFound:
            # Container was removed
            self.container = None
            return False
        except Exception as e:
            logging.warning(f"Failed to check container status: {e}")
            return False

    def _restart_container(self) -> bool:
        """Restart the container with the same image"""
        if not self.current_image:
            logging.error("Cannot restart container: no image name stored")
            return False

        logging.warning(
            f"Container stopped unexpectedly, attempting to restart with image {self.current_image}"
        )
        return self.start_container(self.current_image)

    def execute_command(self, command: str) -> ExecutionResult:
        """Execute a command in the container"""
        if not self.container:
            return ExecutionResult(False, "", "No container running", -1)

        # Check if container is still running, restart if necessary
        if not self._is_container_running():
            logging.warning("Container is not running, attempting to restart...")
            if not self._restart_container():
                return ExecutionResult(
                    False, "", "Failed to restart stopped container", -1
                )

        try:
            # Execute command
            exec_result = self.container.exec_run(
                command,
                stdout=True,
                stderr=True,
                tty=False,
                workdir="/home",
                environment={"TERM": "xterm-256color"},
            )

            return ExecutionResult(
                success=exec_result.exit_code == 0,
                stdout=exec_result.output.decode("utf-8", errors="replace"),
                stderr="",  # stderr is combined with stdout in exec_run
                exit_code=exec_result.exit_code,
            )

        except docker.errors.APIError as e:
            if "is not running" in str(e):
                # Container stopped during execution, try to restart and retry once
                logging.warning(
                    "Container stopped during command execution, attempting restart and retry..."
                )
                if self._restart_container():
                    try:
                        exec_result = self.container.exec_run(
                            command,
                            stdout=True,
                            stderr=True,
                            tty=False,
                            workdir="/home",
                            environment={"TERM": "xterm-256color"},
                        )

                        return ExecutionResult(
                            success=exec_result.exit_code == 0,
                            stdout=exec_result.output.decode("utf-8", errors="replace"),
                            stderr="",  # stderr is combined with stdout in exec_run
                            exit_code=exec_result.exit_code,
                        )
                    except Exception as retry_e:
                        logging.error(
                            f"Failed to execute command after restart: {retry_e}"
                        )
                        return ExecutionResult(
                            False,
                            "",
                            f"Command failed after restart: {str(retry_e)}",
                            -1,
                        )
                else:
                    return ExecutionResult(
                        False, "", "Container stopped and restart failed", -1
                    )
            else:
                logging.error(f"Failed to execute command '{command}': {e}")
                return ExecutionResult(False, "", str(e), -1)
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
                # Keep current_image for potential future restarts
