"""Push a quantized model from the Volume to Hugging Face, with its rendered model card.

Only run after Karthik has approved the push. Needs the Modal secret `huggingface` (HF_TOKEN).

uv run modal run -m quant.push --method awq            # private repo
uv run modal run -m quant.push --method awq --public
"""

import modal

from quant import config
from quant.common import VOLUMES, app
from quant.model_card import gguf_filename, load_record, render

push_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub[hf_xet]")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("quant")
)


@app.function(image=push_image, volumes=VOLUMES, secrets=[modal.Secret.from_name("huggingface")],
              cpu=2, memory=4096, timeout=60 * 60)
def push(method: str, card: str, private: bool, allow_patterns: list[str] | None) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    repo = config.HF_REPOS[method]
    folder = f"{config.MODELS_DIR}/{config.OUTPUT_NAMES[method]}"
    api.create_repo(repo, private=private, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=folder, allow_patterns=allow_patterns,
                      commit_message=f"upload {config.OUTPUT_NAMES[method]}")
    api.upload_file(repo_id=repo, path_or_fileobj=card.encode(), path_in_repo="README.md",
                    commit_message="add model card")
    return f"https://huggingface.co/{repo}"


@app.local_entrypoint()
def run_push(method: str, public: bool = False):
    rec = load_record(method)
    # GGUF: ship the quantized file only, not the F16 intermediate.
    allow = [gguf_filename(rec)] if method == "gguf" else None
    print(push.remote(method, render(rec), private=not public, allow_patterns=allow))
