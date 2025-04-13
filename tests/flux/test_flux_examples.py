import os
import subprocess

import pytest

EXAMPLES_DIR = "./examples"

example_scripts = [f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".py") and f.startswith("flux")]


@pytest.mark.parametrize("script_name", example_scripts)
def test_example_script_runs(script_name):
    script_path = os.path.join(EXAMPLES_DIR, script_name)
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    print(f"Running {script_path} -> Return code: {result.returncode}")
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"{script_path} failed with code {result.returncode}"
