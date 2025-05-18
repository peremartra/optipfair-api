# tests/conftest.py
import os
import sys 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force matplotlib to use the Agg backend (no GUI) during tests
os.environ["MPLBACKEND"] = "Agg"