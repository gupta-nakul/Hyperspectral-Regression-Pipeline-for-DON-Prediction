import os
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

def load_config():
    """Loads and returns the configuration from config.yaml."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
    
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

config = load_config()
