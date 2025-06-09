#!/usr/bin/env python3
"""
Script to check code formatting with Black and isort.
Can be used in CI/CD or as a pre-commit check.
"""

import subprocess
import sys


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"Running {description}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(e.stdout)
        print(e.stderr)
        return False


def main() -> None:
    """Check code formatting with Black and isort."""
    success = True

    # Check Black formatting (--check mode doesn't modify files)
    success &= run_command(
        ["black", "--check", "video_asr_summary/", "tests/", "examples/", "scripts/"],
        "Black formatting check",
    )

    # Check isort import sorting (--check-only mode doesn't modify files)
    success &= run_command(
        [
            "isort",
            "--check-only",
            "video_asr_summary/",
            "tests/",
            "examples/",
            "scripts/",
        ],
        "isort import sorting check",
    )

    if success:
        print("\n🎉 All formatting checks passed!")
        sys.exit(0)
    else:
        print("\n💥 Some formatting checks failed. Run 'black .' and 'isort .' to fix.")
        sys.exit(1)


if __name__ == "__main__":
    main()
