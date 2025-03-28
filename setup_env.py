import sys
import os
from pathlib import Path

def setup_environment():
    """Add the project root directory to Python's path."""
    # Get the current working directory path
    current_dir = os.path.abspath(os.getcwd())
    
    # Determine if we're in the project root or a subdirectory like 'notebooks'
    if os.path.basename(current_dir) == 'notebooks' or 'notebooks' in current_dir:
        # If we're in 'notebooks', go up one level to get to the project root
        project_root = os.path.dirname(current_dir)
    elif os.path.basename(current_dir) == 'Thesis':
        # If we're already at the project root
        project_root = current_dir
    else:
        # Handle other subdirectories by finding 'Thesis' in the path
        path_parts = current_dir.split(os.sep)
        try:
            thesis_index = path_parts.index('Thesis')
            project_root = os.sep.join(path_parts[:thesis_index+1])
        except ValueError:
            # Fallback: just use the current directory
            project_root = current_dir
    
    print(f"Project root: {project_root}")
    
    # Add to the Python path if it's not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added {project_root} to Python path")
    
    return project_root

if __name__ == "__main__":
    setup_environment()