import os
import json
from chronos import ChronosPipeline


def download_pretrained_model_if_not_exists(directory_path, model_name):
    ChronosPipeline.from_pretrained(f"amazon/{model_name}", cache_dir=directory_path)
    print(f"Model `{model_name}` is ready.")


if __name__ == "__main__":
    # relative path to config file
    config_file_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "model_config.json"
    )

    with open(config_file_path) as f:
        model_config = json.load(f)
        model_name = model_config["model_name"]

    pretrained_model_root_path = os.path.join(
        os.path.dirname(__file__), "pretrained_model", model_name
    )
    download_pretrained_model_if_not_exists(pretrained_model_root_path, model_name)
