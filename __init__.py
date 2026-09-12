import importlib
import pkgutil
import sys
from pathlib import Path

__repo_name__ = "ComfyUI-QwenASR"
__version__ = "1.1.0"

# Locate current directory
current_dir = Path(__file__).parent

# Ensure current directory is in sys.path
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"


def load_nodes():
    """Automatically discover and load node definitions."""
    for (_, module_name, _) in pkgutil.iter_modules([str(current_dir)]):
        if not module_name.startswith("AILab_"):
            continue
        try:
            rel_import = f".{module_name}" if __package__ else module_name
            module = importlib.import_module(rel_import, package=__package__)
            if hasattr(module, "NODE_CLASS_MAPPINGS"):
                NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
        except Exception as e:
            print(f"[{__repo_name__}] Error loading {module_name}: {e}")


# Load all nodes
load_nodes()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print(f'\033[36m[{__repo_name__}]\033[0m v'
      f'\033[93m{__version__}\033[0m | '
      f'\033[37m{len(NODE_CLASS_MAPPINGS)} nodes\033[0m '
      f'\033[92mLoaded\033[0m')
