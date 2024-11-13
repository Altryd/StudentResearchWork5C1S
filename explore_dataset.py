import numpy as np
import mimetypes
import cv2
import os


def scantree(path, recursive=False):
    """Recursively yield DirEntry objects for given directory if the "recursive" flag is set to True."""
    for entry in os.scandir(path):
        if recursive and entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path, recursive=True)
        elif entry.is_file(follow_symlinks=False):
            yield entry
        else:
            continue


def get_statistics(path, recursive=False):
    minimal_height = np.inf
    minimal_width = np.inf
    max_height = 0
    max_width = 0
    processed_files = 0
    # with os.scandir(path_to_dir) as iterator:
    for entry in scantree(path, recursive=recursive):
        # allowed_mime_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/']
        allowed_mime_types = ['image/']
        file_mime_type = mimetypes.guess_type(entry.path)[0]
        if not file_mime_type:
            continue
        file_mime_type_category = file_mime_type.split('/')[0] + '/'
        if file_mime_type not in allowed_mime_types and file_mime_type_category not in allowed_mime_types:
            continue
        if entry.is_file():
            image = cv2.imread(entry.path)
            image_width, image_height, channels = image.shape
            if image_width < minimal_width:
                minimal_width = image_width
            if image_width > max_width:
                max_width = image_width
            if image_height < minimal_height:
                minimal_height = image_height
            if image_height > max_height:
                max_height = image_height
            processed_files += 1
        #if entry.is_file() and filetype.guess(entry.path) and filetype.guess(entry.path).MIME in allowed_mime_types:
        #    image_names.append(entry.path)
    return minimal_height, minimal_width, max_height, max_width, processed_files


minimal_height, minimal_width, max_height, max_width, processed_files = get_statistics(
    r'C:\Users\Altryd\CEDAR\signatures', recursive=True)
print(f"Stats: processed_files={processed_files};  min_height={minimal_height}, min_width={minimal_width}, "
      f"max_height={max_height}, max_width={max_width}")
