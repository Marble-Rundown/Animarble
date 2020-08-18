import os


def create_unique_filename(filename):
    """Takes a base filename and constructs a new one if necessary to avoid conflicts"""
    file_base, ext = os.path.splitext(filename)

    destination = f"{filename}"
    n = 0
    while os.path.isfile(destination):
        n += 1
        destination = f"{file_base}({n}){ext}"
    return destination
