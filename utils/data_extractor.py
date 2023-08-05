import os
import Config as Config


def get_files_in_directory():
    files_in_directory = os.listdir(Config.NORMAL_VIDEO_DIRECTORY)
    return files_in_directory
