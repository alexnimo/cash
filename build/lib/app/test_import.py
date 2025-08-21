import sys
import os

print("Python Path:")
for path in sys.path:
    print(f"  - {path}")

print("\nCurrent Working Directory:", os.getcwd())

try:
    import app
    print("\nSuccessfully imported app module")
except ImportError as e:
    print("\nFailed to import app module:", str(e))
