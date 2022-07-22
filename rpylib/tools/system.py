"""Generic tools around handling of files system."""

import os


def get_path_filename(file_path: str) -> tuple[str, str, str]:
    """
    :param file_path: this should be equal to __file__ in most case
    :return: path, file name and extension of the given file path
    """
    path, file = os.path.split(file_path)
    filename, extension = os.path.splitext(file)
    return path, filename, extension


def create_folder(folder_name: str) -> None:
    """Create new folder
    :param folder_name: name of the folder to be created
    """
    os.makedirs(folder_name, exist_ok=True)
