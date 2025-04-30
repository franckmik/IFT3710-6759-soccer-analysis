from pathlib import Path
from typing import List, Union


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory"""
    current_file = Path(__file__).resolve()
    # Navigate up until finding the project root
    for parent in [current_file, *current_file.parents]:
        if (parent / "event detection project").exists():
            return parent
    raise FileNotFoundError("Could not find project root directory")


def get_project_path(*path_parts: str) -> Path:
    """
    Get the absolute path to any file or directory in the project

    Args:
        *path_parts: Path components relative to the project root
                     e.g., "event detection project", "cards", "card_classifier.py"

    Returns:
        Path: Absolute path to the requested file or directory
    """
    project_root = get_project_root()
    return project_root.joinpath(*path_parts)



def get_file_path(relative_path: Union[str, List[str]]) -> Path:
    """
    Get absolute path to any file using either a path string or list of path components

    Args:
        relative_path: Either a string like "event detection project", "cards", "card_classifier.py"
                      or a list like ["backend", "app", "config.json"]

    Returns:
        Path: Absolute path to the file
    """
    if isinstance(relative_path, str):
        parts = relative_path.split('/')
    else:
        parts = relative_path

    return get_project_path(*parts)


