import subprocess, sys, pathlib
py = sys.executable
test = str(pathlib.Path(__file__).parent / "_test_sprint2.py")
result = subprocess.run([py, test], capture_output=True, text=True, encoding="utf-8", errors="replace")
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
sys.exit(result.returncode)
