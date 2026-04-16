"""Directory listing tool for the docchat agent."""
import os
from tools import is_path_safe


def ls(path='.'):
    """
    List files in a directory sorted asciibetically and return as a string.

    >>> ls('test_data')
    'binary.bin\\nhello.txt\\nnumbers.txt\\ntest.png\\nutf16.txt'
    >>> ls('/etc')
    'Error: path is not safe'
    >>> ls('../secret')
    'Error: path is not safe'
    >>> ls('nonexistent_xyz_dir')
    'Error: No such file or directory: nonexistent_xyz_dir'
    """
    if not is_path_safe(path):
        return 'Error: path is not safe'
    try:
        files = sorted(os.listdir(path))
        return '\n'.join(files)
    except FileNotFoundError:
        return f'Error: No such file or directory: {path}'
