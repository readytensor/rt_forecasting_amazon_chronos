import os
from config import paths
from logger import get_logger


logger = get_logger(task_name="train")

def run_training() -> None:
    """
    No-Op
    """
    logger.info("This is a pretrained model, no training required.")
    dummy_file_path = os.path.join(paths.MODEL_ARTIFACTS_PATH, "no_op.txt")
    os.makedirs(paths.MODEL_ARTIFACTS_PATH, exist_ok=True)
    with open(dummy_file_path, "w") as file:
        file.write("Pretrained LLM model. No training required.")


if __name__ == "__main__":
    run_training()
