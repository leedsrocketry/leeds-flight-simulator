import sys
import os

# Ensure the repo root is on sys.path so that `import atmosphere` etc. work
# when pytest is run from any working directory.
sys.path.insert(0, os.path.dirname(__file__))
