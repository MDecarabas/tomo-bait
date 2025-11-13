"""
A module for cloning and updating a git repository, and then building its
Sphinx documentation.
"""

import subprocess
import sys
from pathlib import Path
from typing import Union

from git import Repo


def ingest_documentation(repo_url: str, documentation_dir: Union[str, Path]):
    """
    Clones a repository if it doesn't exist, or pulls the latest changes if it does.
    Then, it builds the Sphinx documentation.

    Args:
        repo_url (str): The URL of the git repository to clone.
        documentation_dir (Union[str, Path]): The path to the directory where the
            documentation and repository will be stored.
    """
    documentation_dir = Path(documentation_dir)
    repo_dir = documentation_dir / "2bm-docs"

    # --- 1. Clone or Pull Repository ---
    if not repo_dir.exists():
        print(f"Cloning repository to: {repo_dir}")
        try:
            Repo.clone_from(repo_url, repo_dir)
        except Exception as e:
            print(f"❌ ERROR: Cloning failed: {e}")
            sys.exit(1)
    else:
        print(f"Pulling latest changes in repository: {repo_dir}")
        try:
            repo = Repo(repo_dir)
            origin = repo.remotes.origin
            origin.pull()
        except Exception as e:
            print(f"❌ ERROR: Pulling failed: {e}")
            sys.exit(1)

    # --- 2. Build Sphinx Documentation ---
    docs_path = repo_dir / "docs"
    if not docs_path.exists():
        print(f"❌ ERROR: 'docs' directory not found in repository: {docs_path}")
        sys.exit(1)

    # It's better to run sphinx-build from the original working directory
    # and specify the source and output directories.
    # This avoids issues with `os.chdir`.
    output_dir = docs_path / "_build"
    command = [
        "sphinx-build",
        "-b",
        "html",  # Build HTML
        str(docs_path),  # Source directory
        str(output_dir),  # Output directory
    ]

    print(f"Running Sphinx build: {' '.join(command)}")

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ Sphinx build successful. Output in: {output_dir}")

    except FileNotFoundError:
        print("❌ ERROR: 'sphinx-build' command not found.")
        print("Please make sure Sphinx is installed in your Python environment.")
        sys.exit(1)

    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Sphinx build failed with code {e.returncode}.")
        print("\n--- Sphinx Output (stdout) ---")
        print(e.stdout)
        print("\n--- Sphinx Errors (stderr) ---")
        print(e.stderr)
        sys.exit(1)


if __name__ == "__main__":
    REPO_URL = "https://github.com/xray-imaging/2bm-docs.git"
    # Store documentation in a 'tomo_documentation' folder in the user's home directory
    DOCS_DIR = Path.home() / "tomo_documentation"
    ingest_documentation(REPO_URL, DOCS_DIR)
