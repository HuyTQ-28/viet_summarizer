import modal
import sys
from pathlib import Path

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/root")
)


app = modal.App("vietnews-summarization", image=image)

volume = modal.Volume.from_name("vietnews_data")

VOLUME_PATH = "/data"

@app.function(
    gpu="A100",
    volumes={str(VOLUME_PATH): volume},
    timeout=86400,
)
def train_model():
    from train import main as training_main
    from utils.config import get_config

    original_get_config = get_config
    def get_modal_config():
        print(f"Loading config with base path: {VOLUME_PATH}")
        return original_get_config(base_path=VOLUME_PATH)

    setattr(sys.modules["utils.config"], "get_config", get_modal_config)
    import train
    train.get_config = get_modal_config
    
    training_main()

if __name__ == "__main__":
    with app.run():
        train_model.remote()