import os

def get_latest_file(dir_path: str, pattern: str, required: bool = True) -> str:
    files = [f for f in os.listdir(dir_path) if f.startswith(pattern)]
    if not files:
        if required:
            raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {dir_path}")
        return None
    return os.path.join(dir_path, max(files, key=lambda x: os.path.getmtime(os.path.join(dir_path, x))))
