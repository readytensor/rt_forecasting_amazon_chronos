import os
from config import paths


def run_training() -> None:
    """
    No-Op
    """
    dummy_file_path = os.path.join(paths.MODEL_ARTIFACTS_PATH, "dummy.txt")
    os.makedirs(paths.MODEL_ARTIFACTS_PATH, exist_ok=True)
    with open(dummy_file_path, "w") as file:
        file.write("model artifacts")


if __name__ == "__main__":
    run_training()
