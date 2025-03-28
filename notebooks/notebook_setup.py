import sys
import os

def setup_notebook_env():
    """Set up the Python path for notebooks to access project modules."""
    # Get the current working directory
    cwd = os.path.abspath(os.getcwd())
    
    # Get the project root directory (parent of notebooks)
    if os.path.basename(cwd) == 'notebooks' or 'notebooks' in cwd:
        project_root = os.path.dirname(cwd)
    else:
        # Look for Thesis in path
        path_parts = cwd.split(os.sep)
        try:
            thesis_index = path_parts.index('Thesis')
            project_root = os.sep.join(path_parts[:thesis_index+1])
        except ValueError:
            project_root = os.path.dirname(cwd)
    
    # Add the project root to sys.path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added {project_root} to Python path")
    
    # Print confirmation message
    print(f"Project root: {project_root}")
    print(f"You can now import project modules (e.g., 'from src.config import CONFIG')")
    
    return project_root