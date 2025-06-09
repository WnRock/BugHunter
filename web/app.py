"""
BugHunter Web Interface - Streamlit Frontend
An intuitive web interface for configuring and running the BugHunter pipeline.
"""

import sys
import yaml
import json
import time
import queue
import threading
import subprocess
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


# Page configuration
st.set_page_config(
    page_title="BugHunter Web Interface",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .log-container {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.375rem;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.875rem;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""",
    unsafe_allow_html=True,
)


class BugHunterWebApp:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_file = self.project_root / "config.yaml"
        self.test_data_file = self.project_root / "data" / "test_set.yaml"

        # Initialize session state
        if "config" not in st.session_state:
            self.load_default_config()
        if "running" not in st.session_state:
            st.session_state.running = False
        if "logs" not in st.session_state:
            st.session_state.logs = []
        if "process" not in st.session_state:
            st.session_state.process = None
        if "log_queue" not in st.session_state:
            st.session_state.log_queue = queue.Queue()

    def load_default_config(self):
        """Load default configuration from config.yaml"""
        try:
            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    st.session_state.config = yaml.safe_load(f)
            else:
                st.session_state.config = self.get_default_config()
        except Exception as e:
            st.error(f"Error loading config: {e}")
            st.session_state.config = self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "tasks": {
                "task_type": "fix_bug",
                "test_data_file": "data/test_set.yaml",
                "target_instances": [],
            },
            "output": {
                "output_dir": "output",
                "run_log": "run.log",
                "trajectory": "trajectory.json",
                "result": "result.txt",
            },
            "model": {
                "name": "DeepSeek-V3",
                "base_url": "https://api.deepseek.com/v1",
                "api_key": "{OPENAI_API_KEY}",
                "generation": {"temperature": 0.1, "max_tokens": 4000},
            },
            "system": {
                "max_iterations": 50,
                "timeout_seconds": 300,
                "log_level": "INFO",
                "num_workers": 1,
            },
            "docker": {
                "pull_timeout": 600,
                "execution_timeout": 120,
                "cleanup_on_exit": True,
                "network_mode": "bridge",
            },
            "prompts": {
                "directory": "bughunter/prompts",
                "fix_bug": "fix_bug.txt",
                "locate_bug": "locate_bug.txt",
                "fix_with_location": "fix_with_location.txt",
            },
        }

    def load_test_instances(self) -> List[Dict[str, Any]]:
        """Load available test instances"""
        try:
            if self.test_data_file.exists():
                with open(self.test_data_file, "r") as f:
                    return yaml.safe_load(f) or []
            return []
        except Exception as e:
            st.error(f"Error loading test instances: {e}")
            return []

    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, "w") as f:
                yaml.dump(
                    st.session_state.config, f, default_flow_style=False, indent=2
                )
            return True
        except Exception as e:
            st.error(f"Error saving config: {e}")
            return False

    def render_header(self):
        """Render the main header"""
        st.markdown(
            '<h1 class="main-header">🐛 BugHunter Web Interface</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            "An automated pipeline that uses LLM agents to solve software issues in Docker containers."
        )

    def render_sidebar(self):
        """Render the sidebar with navigation"""
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page:",
            [
                "🏠 Dashboard",
                "⚙️ Configuration",
                "🚀 Run Pipeline",
                "📊 Results",
                "📋 Logs",
                "❓ Help",
            ],
        )
        return page

    def render_configuration_page(self):
        """Render the configuration page"""
        st.header("⚙️ Configuration")

        config = st.session_state.config

        # Task Configuration
        st.subheader("📋 Task Configuration")
        col1, col2 = st.columns(2)

        with col1:
            task_type = st.selectbox(
                "Task Type",
                options=["fix_bug", "locate_bug", "fix_with_location"],
                index=["fix_bug", "locate_bug", "fix_with_location"].index(
                    config["tasks"]["task_type"]
                ),
                help="Type of task to perform",
            )
            config["tasks"]["task_type"] = task_type

        with col2:
            test_data_file = st.text_input(
                "Test Data File",
                value=config["tasks"]["test_data_file"],
                help="Path to YAML file containing test instances",
            )
            config["tasks"]["test_data_file"] = test_data_file

        # Target Instances Selection
        st.subheader("🎯 Target Instances")
        test_instances = self.load_test_instances()
        if test_instances:
            instance_options = [inst["instance_id"] for inst in test_instances]
            selected_instances = st.multiselect(
                "Select instances to run (leave empty for all)",
                options=instance_options,
                default=config["tasks"].get("target_instances", []),
                help="Select specific instances to run, or leave empty to run all",
            )
            config["tasks"]["target_instances"] = selected_instances

            # Show instance details
            if st.checkbox("Show instance details"):
                for inst in test_instances:
                    if (
                        not selected_instances
                        or inst["instance_id"] in selected_instances
                    ):
                        with st.expander(f"📄 {inst['instance_id']}"):
                            st.write(f"**Image:** {inst['image_name']}")
                            st.write(f"**Problem:**")
                            st.text(
                                inst["problem_statement"][:500] + "..."
                                if len(inst["problem_statement"]) > 500
                                else inst["problem_statement"]
                            )
        else:
            st.warning("No test instances found. Please check the test data file path.")

        # Model Configuration
        st.subheader("🤖 Model Configuration")
        col1, col2 = st.columns(2)

        with col1:
            model_name = st.text_input(
                "Model Name",
                value=config["model"]["name"],
                help="Name of the LLM model to use",
            )
            config["model"]["name"] = model_name

            base_url = st.text_input(
                "Base URL",
                value=config["model"]["base_url"],
                help="API base URL for the model",
            )
            config["model"]["base_url"] = base_url

        with col2:
            api_key = st.text_input(
                "API Key",
                value=config["model"]["api_key"],
                type=(
                    "password"
                    if not config["model"]["api_key"].startswith("{")
                    else "default"
                ),
                help="API key for the model (use {ENV_VAR} for environment variables)",
            )
            config["model"]["api_key"] = api_key

        # Generation Parameters
        st.subheader("🎛️ Generation Parameters")
        col1, col2 = st.columns(2)

        with col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=float(config["model"]["generation"]["temperature"]),
                step=0.1,
                help="Controls randomness in generation",
            )
            config["model"]["generation"]["temperature"] = temperature

        with col2:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=8000,
                value=config["model"]["generation"]["max_tokens"],
                step=100,
                help="Maximum tokens to generate",
            )
            config["model"]["generation"]["max_tokens"] = max_tokens

        # System Configuration
        st.subheader("⚡ System Configuration")
        col1, col2, col3 = st.columns(3)

        with col1:
            max_iterations = st.number_input(
                "Max Iterations",
                min_value=1,
                max_value=100,
                value=config["system"]["max_iterations"],
                help="Maximum iterations per task",
            )
            config["system"]["max_iterations"] = max_iterations

        with col2:
            timeout_seconds = st.number_input(
                "Timeout (seconds)",
                min_value=60,
                max_value=3600,
                value=config["system"]["timeout_seconds"],
                help="Timeout for each task",
            )
            config["system"]["timeout_seconds"] = timeout_seconds

        with col3:
            num_workers = st.number_input(
                "Number of Workers",
                min_value=1,
                max_value=10,
                value=config["system"]["num_workers"],
                help="Number of parallel workers",
            )
            config["system"]["num_workers"] = num_workers

        log_level = st.selectbox(
            "Log Level",
            options=["DEBUG", "INFO", "WARNING", "ERROR"],
            index=["DEBUG", "INFO", "WARNING", "ERROR"].index(
                config["system"]["log_level"]
            ),
            help="Logging level",
        )
        config["system"]["log_level"] = log_level

        # Output Configuration
        st.subheader("📁 Output Configuration")
        col1, col2 = st.columns(2)

        with col1:
            output_dir = st.text_input(
                "Output Directory",
                value=config["output"]["output_dir"],
                help="Directory to save results",
            )
            config["output"]["output_dir"] = output_dir

        with col2:
            run_log = st.text_input(
                "Run Log Filename",
                value=config["output"]["run_log"],
                help="Filename for run logs",
            )
            config["output"]["run_log"] = run_log

        # Docker Configuration
        st.subheader("🐳 Docker Configuration")
        col1, col2 = st.columns(2)

        with col1:
            pull_timeout = st.number_input(
                "Pull Timeout (seconds)",
                min_value=60,
                max_value=1800,
                value=config["docker"]["pull_timeout"],
                help="Timeout for Docker image pulls",
            )
            config["docker"]["pull_timeout"] = pull_timeout

            execution_timeout = st.number_input(
                "Execution Timeout (seconds)",
                min_value=30,
                max_value=600,
                value=config["docker"]["execution_timeout"],
                help="Timeout for Docker command execution",
            )
            config["docker"]["execution_timeout"] = execution_timeout

        with col2:
            cleanup_on_exit = st.checkbox(
                "Cleanup on Exit",
                value=config["docker"]["cleanup_on_exit"],
                help="Clean up Docker containers on exit",
            )
            config["docker"]["cleanup_on_exit"] = cleanup_on_exit

            network_mode = st.selectbox(
                "Network Mode",
                options=["bridge", "host", "none"],
                index=["bridge", "host", "none"].index(
                    config["docker"]["network_mode"]
                ),
                help="Docker network mode",
            )
            config["docker"]["network_mode"] = network_mode

        # Save Configuration
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("💾 Save Configuration", type="primary"):
                if self.save_config():
                    st.success("Configuration saved successfully!")
                    time.sleep(1)
                    st.rerun()

    def render_dashboard(self):
        """Render the main dashboard"""
        st.header("🏠 Dashboard")

        # Status Overview
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Task Type", st.session_state.config["tasks"]["task_type"])

        with col2:
            test_instances = self.load_test_instances()
            target_instances = st.session_state.config["tasks"].get(
                "target_instances", []
            )
            instance_count = (
                len(target_instances) if target_instances else len(test_instances)
            )
            st.metric("Instances to Run", instance_count)

        with col3:
            st.metric("Model", st.session_state.config["model"]["name"])

        with col4:
            st.metric("Workers", st.session_state.config["system"]["num_workers"])

        # Quick Actions
        st.subheader("🚀 Quick Actions")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("⚙️ Setup System", help="Run system setup and validation"):
                self.run_setup()

        with col2:
            if st.button("🚀 Run Pipeline", help="Start the BugHunter pipeline"):
                st.switch_page("🚀 Run Pipeline")

        with col3:
            if st.button("📊 View Results", help="View recent results"):
                st.switch_page("📊 Results")

        # Recent Activity
        st.subheader("📈 Recent Activity")
        self.show_recent_results()

        # System Information
        st.subheader("ℹ️ System Information")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Project Root:**", str(self.project_root))
            st.write("**Config File:**", str(self.config_file))
            st.write(
                "**Output Directory:**", st.session_state.config["output"]["output_dir"]
            )

        with col2:
            # Check Docker availability
            try:
                result = subprocess.run(
                    ["docker", "--version"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    st.success("✅ Docker is available")
                    st.code(result.stdout.strip())
                else:
                    st.error("❌ Docker is not available")
            except Exception:
                st.error("❌ Docker is not available")

    def run_setup(self):
        """Run the BugHunter setup"""
        try:
            with st.spinner("Running setup..."):
                result = subprocess.run(
                    [sys.executable, str(self.project_root / "main.py"), "setup"],
                    capture_output=True,
                    text=True,
                    cwd=str(self.project_root),
                    timeout=60,
                )

            if result.returncode == 0:
                st.success("✅ Setup completed successfully!")
                st.code(result.stdout)
            else:
                st.error("❌ Setup failed!")
                st.code(result.stderr)

        except subprocess.TimeoutExpired:
            st.error("⏰ Setup timed out!")
        except Exception as e:
            st.error(f"❌ Setup error: {e}")

    def show_recent_results(self):
        """Show recent results from the output directory"""
        output_dir = (
            Path(self.project_root) / st.session_state.config["output"]["output_dir"]
        )

        if not output_dir.exists():
            st.info("No results found. Run the pipeline to generate results.")
            return

        # Look for results.json
        results_file = output_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    results = json.load(f)

                if results:
                    df_data = []
                    for result in results[-10:]:  # Show last 10 results
                        df_data.append(
                            {
                                "Instance ID": result.get("instance_id", "Unknown"),
                                "Task Type": result.get("task_type", "Unknown"),
                                "Success": (
                                    "✅" if result.get("success", False) else "❌"
                                ),
                                "Iterations": result.get("iterations", 0),
                                "Error": (
                                    result.get("error", "")[:50] + "..."
                                    if result.get("error")
                                    else ""
                                ),
                            }
                        )

                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No results found in results.json")

            except Exception as e:
                st.error(f"Error loading results: {e}")
        else:
            st.info("No results.json found. Run the pipeline to generate results.")

    def render_run_page(self):
        """Render the pipeline execution page"""
        st.header("🚀 Run Pipeline")

        # Configuration Summary
        st.subheader("📋 Current Configuration")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(f"**Task Type:** {st.session_state.config['tasks']['task_type']}")
            st.write(f"**Model:** {st.session_state.config['model']['name']}")

        with col2:
            test_instances = self.load_test_instances()
            target_instances = st.session_state.config["tasks"].get(
                "target_instances", []
            )
            instance_count = (
                len(target_instances) if target_instances else len(test_instances)
            )
            st.write(f"**Instances:** {instance_count}")
            st.write(f"**Workers:** {st.session_state.config['system']['num_workers']}")

        with col3:
            st.write(
                f"**Max Iterations:** {st.session_state.config['system']['max_iterations']}"
            )
            st.write(
                f"**Output Dir:** {st.session_state.config['output']['output_dir']}"
            )

        # Pipeline Controls
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if not st.session_state.running:
                if st.button(
                    "🚀 Start Pipeline", type="primary", use_container_width=True
                ):
                    self.start_pipeline()
            else:
                if st.button(
                    "⏹️ Stop Pipeline", type="secondary", use_container_width=True
                ):
                    self.stop_pipeline()

        # Status Display
        if st.session_state.running:
            st.info("🔄 Pipeline is running...")
        elif st.session_state.process and st.session_state.process.poll() is not None:
            if st.session_state.process.returncode == 0:
                st.success("✅ Pipeline completed successfully!")
            else:
                st.error("❌ Pipeline failed!")

        # Real-time Logs
        st.subheader("📋 Real-time Logs")
        log_container = st.container()

        # Auto-refresh logs
        if st.session_state.running:
            # Update logs from queue
            self.update_logs_from_queue()

            with log_container:
                if st.session_state.logs:
                    log_text = "\n".join(
                        st.session_state.logs[-50:]
                    )  # Show last 50 lines
                    st.markdown(
                        f'<div class="log-container">{log_text}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("Waiting for logs...")

            # Auto-refresh every 2 seconds
            time.sleep(2)
            st.rerun()
        else:
            with log_container:
                if st.session_state.logs:
                    log_text = "\n".join(st.session_state.logs[-50:])
                    st.markdown(
                        f'<div class="log-container">{log_text}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No logs available. Start the pipeline to see logs.")

    def start_pipeline(self):
        """Start the BugHunter pipeline"""
        try:
            # Save current config before starting
            if not self.save_config():
                st.error("Failed to save configuration!")
                return

            # Clear previous logs
            st.session_state.logs = []

            # Start the pipeline process
            cmd = [sys.executable, str(self.project_root / "main.py"), "run"]

            st.session_state.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=str(self.project_root),
            )

            st.session_state.running = True

            # Start log monitoring thread
            log_thread = threading.Thread(target=self.monitor_logs, daemon=True)
            log_thread.start()

            st.success("🚀 Pipeline started!")

        except Exception as e:
            st.error(f"❌ Failed to start pipeline: {e}")

    def stop_pipeline(self):
        """Stop the running pipeline"""
        try:
            if st.session_state.process:
                st.session_state.process.terminate()
                st.session_state.process.wait(timeout=10)
                st.session_state.process = None

            st.session_state.running = False
            st.warning("⏹️ Pipeline stopped!")

        except Exception as e:
            st.error(f"❌ Failed to stop pipeline: {e}")

    def monitor_logs(self):
        """Monitor pipeline logs in a separate thread"""
        try:
            while st.session_state.running and st.session_state.process:
                line = st.session_state.process.stdout.readline()
                if line:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    formatted_line = f"[{timestamp}] {line.strip()}"
                    st.session_state.log_queue.put(formatted_line)
                elif st.session_state.process.poll() is not None:
                    # Process has ended
                    st.session_state.running = False
                    break

        except Exception as e:
            st.session_state.log_queue.put(f"[ERROR] Log monitoring failed: {e}")
            st.session_state.running = False

    def update_logs_from_queue(self):
        """Update logs from the queue"""
        try:
            while not st.session_state.log_queue.empty():
                log_line = st.session_state.log_queue.get_nowait()
                st.session_state.logs.append(log_line)

                # Keep only last 1000 lines to prevent memory issues
                if len(st.session_state.logs) > 1000:
                    st.session_state.logs = st.session_state.logs[-1000:]

        except queue.Empty:
            pass

    def render_results_page(self):
        """Render the results page"""
        st.header("📊 Results")

        output_dir = (
            Path(self.project_root) / st.session_state.config["output"]["output_dir"]
        )

        if not output_dir.exists():
            st.warning("No output directory found. Run the pipeline first.")
            return

        # Main results file
        results_file = output_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    results = json.load(f)

                if results:
                    # Summary statistics
                    st.subheader("📈 Summary")
                    total_tasks = len(results)
                    successful_tasks = sum(
                        1 for r in results if r.get("success", False)
                    )

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Tasks", total_tasks)
                    with col2:
                        st.metric("Successful", successful_tasks)
                    with col3:
                        st.metric("Failed", total_tasks - successful_tasks)
                    with col4:
                        success_rate = (
                            (successful_tasks / total_tasks * 100)
                            if total_tasks > 0
                            else 0
                        )
                        st.metric("Success Rate", f"{success_rate:.1f}%")

                    # Detailed results table
                    st.subheader("📋 Detailed Results")

                    df_data = []
                    for result in results:
                        df_data.append(
                            {
                                "Instance ID": result.get("instance_id", "Unknown"),
                                "Task Type": result.get("task_type", "Unknown"),
                                "Success": result.get("success", False),
                                "Iterations": result.get("iterations", 0),
                                "Error": result.get("error", ""),
                            }
                        )

                    df = pd.DataFrame(df_data)

                    # Filters
                    col1, col2 = st.columns(2)
                    with col1:
                        task_filter = st.selectbox(
                            "Filter by Task Type",
                            ["All"] + df["Task Type"].unique().tolist(),
                        )
                    with col2:
                        status_filter = st.selectbox(
                            "Filter by Status", ["All", "Success", "Failed"]
                        )

                    # Apply filters
                    filtered_df = df.copy()
                    if task_filter != "All":
                        filtered_df = filtered_df[
                            filtered_df["Task Type"] == task_filter
                        ]
                    if status_filter == "Success":
                        filtered_df = filtered_df[filtered_df["Success"] == True]
                    elif status_filter == "Failed":
                        filtered_df = filtered_df[filtered_df["Success"] == False]

                    st.dataframe(filtered_df, use_container_width=True)

                    # Individual result details
                    st.subheader("🔍 Individual Results")
                    instance_ids = filtered_df["Instance ID"].tolist()

                    if instance_ids:
                        selected_instance = st.selectbox(
                            "Select instance for details:", instance_ids
                        )

                        # Show detailed result for selected instance
                        selected_result = next(
                            (
                                r
                                for r in results
                                if r.get("instance_id") == selected_instance
                            ),
                            None,
                        )
                        if selected_result:
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("**Instance Details:**")
                                st.json(
                                    {
                                        "instance_id": selected_result.get(
                                            "instance_id"
                                        ),
                                        "task_type": selected_result.get("task_type"),
                                        "success": selected_result.get("success"),
                                        "iterations": selected_result.get("iterations"),
                                        "error": selected_result.get("error"),
                                    }
                                )

                            with col2:
                                st.write("**Result Data:**")
                                result_data = selected_result.get("result_data", {})
                                if result_data:
                                    st.json(result_data)
                                else:
                                    st.info("No result data available")

                            # Show trajectory summary if available
                            if "trajectory_summary" in selected_result:
                                st.write("**Trajectory Summary:**")
                                st.json(selected_result["trajectory_summary"])

                else:
                    st.info("No results found in results.json")

            except Exception as e:
                st.error(f"Error loading results: {e}")
        else:
            st.warning("No results.json found. Run the pipeline to generate results.")

        # Instance-specific results
        st.subheader("📁 Instance-specific Results")
        instance_dirs = [d for d in output_dir.iterdir() if d.is_dir()]

        if instance_dirs:
            selected_dir = st.selectbox(
                "Select instance directory:", [d.name for d in instance_dirs]
            )

            if selected_dir:
                instance_dir = output_dir / selected_dir

                # Show files in instance directory
                files = list(instance_dir.glob("*"))
                if files:
                    selected_file = st.selectbox(
                        "Select file to view:", [f.name for f in files]
                    )

                    if selected_file:
                        file_path = instance_dir / selected_file

                        try:
                            if file_path.suffix in [".json", ".txt", ".log"]:
                                with open(file_path, "r") as f:
                                    content = f.read()

                                if file_path.suffix == ".json":
                                    try:
                                        json_data = json.loads(content)
                                        st.json(json_data)
                                    except json.JSONDecodeError:
                                        st.code(content)
                                else:
                                    st.code(content)
                            else:
                                st.info(
                                    f"File type {file_path.suffix} not supported for preview"
                                )

                        except Exception as e:
                            st.error(f"Error reading file: {e}")
                else:
                    st.info(f"No files found in {selected_dir}")
        else:
            st.info("No instance directories found")

    def render_logs_page(self):
        """Render the logs page"""
        st.header("📋 Logs")

        output_dir = (
            Path(self.project_root) / st.session_state.config["output"]["output_dir"]
        )

        # Main pipeline log
        st.subheader("🔍 Main Pipeline Log")
        main_log = output_dir / "bughunter_pipeline.log"

        if main_log.exists():
            try:
                with open(main_log, "r") as f:
                    log_content = f.read()

                # Log level filter
                log_levels = ["ALL", "DEBUG", "INFO", "WARNING", "ERROR"]
                selected_level = st.selectbox("Filter by log level:", log_levels)

                if selected_level != "ALL":
                    lines = log_content.split("\n")
                    filtered_lines = [
                        line
                        for line in lines
                        if selected_level in line
                        or not any(level in line for level in log_levels[1:])
                    ]
                    log_content = "\n".join(filtered_lines)

                st.code(log_content, language="text")

            except Exception as e:
                st.error(f"Error reading main log: {e}")
        else:
            st.info("No main pipeline log found")

        # Instance-specific logs
        st.subheader("📁 Instance Logs")
        instance_dirs = [d for d in output_dir.iterdir() if d.is_dir()]

        if instance_dirs:
            selected_instance = st.selectbox(
                "Select instance:", [d.name for d in instance_dirs]
            )

            if selected_instance:
                instance_dir = output_dir / selected_instance
                run_log = instance_dir / st.session_state.config["output"]["run_log"]

                if run_log.exists():
                    try:
                        with open(run_log, "r") as f:
                            instance_log_content = f.read()
                        st.code(instance_log_content, language="text")
                    except Exception as e:
                        st.error(f"Error reading instance log: {e}")
                else:
                    st.info(f"No run log found for {selected_instance}")
        else:
            st.info("No instance directories found")

    def render_help_page(self):
        """Render the help page"""
        st.header("❓ Help")

        st.markdown(
            """
        ## 🐛 BugHunter Web Interface Help
        
        This web interface provides an intuitive way to configure and run the BugHunter pipeline.
        
        ### 📖 Overview
        
        BugHunter is an automated pipeline that uses LLM agents to solve different types of software issues in Docker containers. The system supports multiple task types and provides comprehensive logging and trajectory recording.
        
        ### 🔧 Task Types
        
        1. **Fix Bug (`fix_bug`)**: Analyze the problem and provide a complete patch to fix the issue
        2. **Locate Bug (`locate_bug`)**: Identify the specific file and line number where the bug is located
        3. **Fix with Location (`fix_with_location`)**: Fix a bug when you already know approximately where it is located
        
        ### 📋 Pages
        
        - **🏠 Dashboard**: Overview of the system status and quick actions
        - **⚙️ Configuration**: Configure all pipeline settings including tasks, model, system, and Docker parameters
        - **🚀 Run Pipeline**: Execute the pipeline and monitor real-time logs
        - **📊 Results**: View and analyze pipeline results with detailed breakdowns
        - **📋 Logs**: View detailed logs from pipeline execution
        - **❓ Help**: This help page
        
        ### 🚀 Quick Start
        
        1. **Configure**: Go to the Configuration page and set up your model API key and other settings
        2. **Setup**: Run the system setup from the Dashboard to validate your environment
        3. **Select Instances**: Choose which test instances to run (or leave empty for all)
        4. **Run**: Start the pipeline from the Run Pipeline page
        5. **Monitor**: Watch real-time logs and progress
        6. **Analyze**: Review results on the Results page
        
        ### ⚙️ Configuration Tips
        
        - **API Key**: Use `{OPENAI_API_KEY}` format to reference environment variables
        - **Workers**: Use multiple workers for parallel processing of instances
        - **Timeout**: Adjust timeout based on the complexity of your tasks
        - **Output Directory**: Results will be saved in subdirectories by instance ID
        
        ### 🐳 Docker Requirements
        
        - Docker must be installed and running
        - Current user must have Docker permissions
        - Internet connection required for pulling Docker images
        
        ### 📊 Understanding Results
        
        - **Success**: Whether the task completed without errors
        - **Iterations**: Number of agent iterations used
        - **Result Data**: Task-specific output (patches, locations, etc.)
        - **Trajectory**: Detailed record of agent interactions
        
        ### 🔍 Troubleshooting
        
        - **Setup Issues**: Check Docker installation and permissions
        - **API Errors**: Verify API key and model configuration
        - **Timeout Errors**: Increase timeout values or reduce task complexity
        - **Memory Issues**: Reduce number of workers or max iterations
        
        ### 📝 Support
        
        For additional support, please refer to the main BugHunter documentation or check the project repository.
        """
        )

    def run(self):
        """Main application runner"""
        self.render_header()

        page = self.render_sidebar()

        # Route to appropriate page
        if page == "🏠 Dashboard":
            self.render_dashboard()
        elif page == "⚙️ Configuration":
            self.render_configuration_page()
        elif page == "🚀 Run Pipeline":
            self.render_run_page()
        elif page == "📊 Results":
            self.render_results_page()
        elif page == "📋 Logs":
            self.render_logs_page()
        elif page == "❓ Help":
            self.render_help_page()


def main():
    """Main entry point"""
    app = BugHunterWebApp()
    app.run()


if __name__ == "__main__":
    main()
