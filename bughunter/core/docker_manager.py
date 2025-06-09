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
                logging.info(f"Image {image_name} found locally")
            except docker.errors.ImageNotFound:
                logging.info(f"Pulling image {image_name}...")
                self.client.images.pull(image_name)

            # Start the container with better process handling
            self.container = self.client.containers.run(
                image_name,
                command=[
                    "sleep",
                    "infinity",
                ],  # Use array form for better process handling
                detach=True,
                working_dir="/home",
                tty=True,
                stdin_open=True,  # Keep STDIN open
                remove=False,  # Don't auto-remove so we can check logs if it fails
            )

            # Store the image name for potential restart
            self.current_image = image_name

            # Wait longer for container to be fully ready and verify it's actually running
            max_wait_time = 10  # Maximum wait time in seconds
            wait_interval = 0.5  # Check every 0.5 seconds
            total_waited = 0

            while total_waited < max_wait_time:
                time.sleep(wait_interval)
                total_waited += wait_interval

                # Check if container is running
                if self._is_container_running():
                    # Additional check: try to execute a simple command to verify the container is responsive
                    try:
                        test_result = self.container.exec_run(
                            "echo 'container_ready'", stdout=True, stderr=True
                        )
                        if test_result.exit_code == 0:
                            logging.info(
                                f"Container {self.container.short_id} started successfully and is responsive"
                            )
                            return True
                    except Exception as e:
                        logging.warning(f"Container responsiveness test failed: {e}")
                        continue  # Continue waiting
                else:
                    # Container stopped, check logs for the error
                    try:
                        logs = self.container.logs().decode("utf-8", errors="replace")
                        logging.error(f"Container failed to start. Logs: {logs}")
                    except:
                        pass
                    return False

            # If we get here, the container didn't become responsive in time
            logging.error(
                f"Container {self.container.short_id} started but is not responsive after {max_wait_time}s"
            )
            try:
                logs = self.container.logs().decode("utf-8", errors="replace")
                logging.error(f"Container logs: {logs}")
            except:
                pass
            return False

        except Exception as e:
            logging.error(f"Failed to start container: {e}")
            # If container was created but failed, try to get its logs
            if self.container:
                try:
                    logs = self.container.logs().decode("utf-8", errors="replace")
                    logging.error(f"Container logs: {logs}")
                except:
                    pass
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
            # Execute command with timeout to prevent hanging
            exec_result = self.container.exec_run(
                command,
                stdout=True,
                stderr=True,
                tty=False,
                workdir="/home",
                environment={"TERM": "xterm-256color"},
                stream=False,  # Don't stream, get all output at once
                demux=False,  # Don't demux stdout/stderr
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
                            stream=False,
                            demux=False,
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
