import modal
import os

# Đường dẫn local
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
TOKENIZER_DIR = os.path.join(ROOT_DIR, "tokenizers")

# Volume
volume = modal.Volume.from_name("vietnews_data", create_if_missing=True)

def upload():
    remote_data_dir = "/data"
    remote_tokenizer_dir = "/tokenizers"

    print("Uploading.............")
    with volume.batch_upload() as batch:
        # Upload train/val/test
        batch.put_file(os.path.join(DATA_DIR, "train.csv"), f"{remote_data_dir}/train.csv")
        batch.put_file(os.path.join(DATA_DIR, "validation.csv"), f"{remote_data_dir}/validation.csv")
        batch.put_file(os.path.join(DATA_DIR, "test.csv"), f"{remote_data_dir}/test.csv")

        # Upload tokenizer
        batch.put_file(
            os.path.join(TOKENIZER_DIR, "spm_vietnamese_model.model"),
            f"{remote_tokenizer_dir}/spm_vietnamese_model.model"
        )

    print("Data uploaded successfully!")

if __name__ == "__main__":
    upload()
