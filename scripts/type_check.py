#!/usr/bin/env python3
"""Script to run type checking with mypy."""

import subprocess
import sys
from pathlib import Path


def main():
    """Run mypy type checking on the video_asr_summary package."""
    project_root = Path(__file__).parent.parent
    package_path = project_root / "video_asr_summary"

    print("Running mypy type checking...")
    result = subprocess.run(
        ["mypy", str(package_path)], cwd=project_root, capture_output=True, text=True
    )

    if result.returncode == 0:
        print("✅ No type errors found!")
    else:
        print("❌ Type errors found:")
        print(result.stdout)
        if result.stderr:
            print("Stderr:")
            print(result.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
