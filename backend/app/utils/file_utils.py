"""
File utility functions for the Financial Intelligence System.

This module contains utility functions for file operations,
document handling, and file system management.
"""

import os
from typing import Union


def save_demo_doc(path: str, contents: Union[str, bytes]) -> None:
    """
    Save demo document to specified path.
    
    Args:
        path: File path to save the document
        contents: Document contents (string or bytes)
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Write contents based on type
    mode = 'wb' if isinstance(contents, bytes) else 'w'
    with open(path, mode) as f:
        f.write(contents)