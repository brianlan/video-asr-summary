#!/usr/bin/env python3
"""Development runner script that sets up Python path automatically."""

import sys
from pathlib import Path

# Add project root to Python path so we can import video_asr_summary directly
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import subprocess
    
    # Run pytest with the configured path
    cmd = [sys.executable, "-m", "pytest"] + sys.argv[1:]
    sys.exit(subprocess.run(cmd).returncode)
