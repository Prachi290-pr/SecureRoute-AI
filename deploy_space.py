import os
from pathlib import Path

from huggingface_hub import HfApi


def as_bool(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    token = os.getenv("HF_TOKEN")
    space_id = os.getenv("HF_SPACE_ID")

    if not token:
        raise RuntimeError("HF_TOKEN is required")
    if not space_id:
        raise RuntimeError("HF_SPACE_ID is required (format: username/space-name)")

    private_space = as_bool(os.getenv("HF_SPACE_PRIVATE"))

    api = HfApi(token=token)
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="docker",
        private=private_space,
        exist_ok=True,
    )

    api.upload_folder(
        folder_path=str(Path(__file__).resolve().parent),
        repo_id=space_id,
        repo_type="space",
        ignore_patterns=[
            ".git/*",
            ".venv/*",
            "__pycache__/*",
            ".secrets/*",
            "*.pyc",
            "*.ipynb",
        ],
    )

    for secret_name in ["HF_TOKEN", "API_BASE_URL", "MODEL_NAME"]:
        secret_value = os.getenv(secret_name)
        if secret_value:
            api.add_space_secret(repo_id=space_id, key=secret_name, value=secret_value)

    print(f"Space deployed: https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    main()
