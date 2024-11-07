import os
from pathlib import Path
from typing import List

import filetype


def resolve_path(file_path: str) -> str:
    return os.path.abspath(file_path)


def is_file(file_path: str) -> bool:
    return bool(file_path and os.path.isfile(file_path))


def is_directory(directory_path: str) -> bool:
    return bool(directory_path and os.path.isdir(directory_path))


def is_image(image_path: str) -> bool:
    return is_file(image_path) and filetype.helpers.is_image(image_path)


def ensure_directory(file_path: str) -> bool:
    parent_dir = Path(file_path).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    return parent_dir.is_dir()


def list_files(directory_path: str) -> List[str]:
    if is_directory(directory_path):
        files = os.listdir(directory_path)
        files = [
            os.path.abspath(os.path.join(directory_path, file))
            for file in files
            if not file.startswith((".", "__"))
        ]
        return sorted(files)
    return []


def list_images(directory_path: str) -> List[str]:
    return [path for path in list_files(directory_path) if is_image(path)]
