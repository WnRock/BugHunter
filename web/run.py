"""
BugHunter Web Interface Launcher
Quick launcher script for the Streamlit web interface
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Launch the BugHunter web interface"""
    # Get the directory of this script
    web_dir = Path(__file__).parent
    project_root = web_dir.parent

    # Launch streamlit
    app_file = web_dir / "app.py"

    print("🚀 Starting BugHunter Web Interface...")
    print(f"📁 Project root: {project_root}")
    print(f"🌐 Web interface will open in your browser")
    print("📝 Use Ctrl+C to stop the server")

    # Run streamlit
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_file),
            "--server.address",
            "0.0.0.0",
            "--server.port",
            "8501",
            "--browser.gatherUsageStats",
            "false",
        ]
    )


if __name__ == "__main__":
    main()
