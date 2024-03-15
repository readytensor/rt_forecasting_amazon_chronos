import os
import requests

def download_pretrained_model_if_not_exists(directory_path, model_name="chronos-t5-tiny"):
    print("Downloading pretrained model...")
    files_urls = {
        "config.json": f"https://huggingface.co/amazon/{model_name}/resolve/main/config.json",
        "generation_config.json": f"https://huggingface.co/amazon/{model_name}/resolve/main/generation_config.json",
        "pytorch_model.bin": f"https://huggingface.co/amazon/{model_name}/resolve/main/pytorch_model.bin",
    }

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for file_name, url in files_urls.items():
        file_path = os.path.join(directory_path, file_name)
        if not os.path.exists(file_path):
            try:
                print(f"Downloading {file_name}...")
                response = requests.get(url, allow_redirects=True)
                response.raise_for_status()  # Raise an HTTPError for bad responses
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            except requests.RequestException as e:
                raise ValueError(f"Error downloading pretrained model file from {file_path}.") from e

    print("Pretrained model downloaded successfully.")


if __name__ == "__main__":
    # Example usage
    directory_path = "."  # Replace with your directory path
    model_name = "chronos-t5-tiny" # choose between tiny, mini, small, base and large
    download_if_not_exists(directory_path, model_name)