import pandas as pd
import re
import os
import unicodedata
from tqdm.auto import tqdm
import html

tqdm.pandas()


# Cấu hình
DATASET_NAME = "ithieund/VietNews-Abs-Sum"
PROCESSED_DATA_DIR = "data/processed"
MIN_ARTICLE_LEN = 50
MIN_SUMMARY_LEN = 10


def normalize_numbers(text):
    text = re.sub(r'(\d+),(\d+)', r'\1.\2', text)
    while re.search(r'(\d+)\.(\d{3})', text):
        text = re.sub(r'(\d+)\.(\d{3})', r'\1\2', text)
    return text

def normalize_dates(text):
    text = re.sub(r'\b(\d{1,2})[-.\s](\d{1,2})[-.\s](\d{2,4})\b', r'\1/\2/\3', text)
    text = re.sub(r'\b(\d{1,2})[-.\s](\d{1,2})\b(?![\d/])', r'\1/\2', text)
    return text

def clean_text(text):
    """Hàm xử lý văn bản"""
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFC", text) # Chuẩn hóa Unicode
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text) # Xóa các ký tự vô hình
    text = text.lower() # Chuyển văn bản thành chữ thường

    # Loại bỏ nhiễu: links, chú thích ảnh/video ---
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s*(ảnh|video)\s*:\s*[\w\s_.-]+', ' ', text)

    # Các bước chuẩn hóa
    text = normalize_dates(text)
    text = normalize_numbers(text)

    # Xử lý dấu câu
    text = re.sub(r'([,!?:;()"])', r' \1 ', text) # Tách dấu câu khỏi từ
    text = re.sub(r'(?<!\d)\.(?!\d)', ' . ', text) # Tách dấu chấm có điều kiện
    
    # Loại bỏ các ký tự không mong muốn, giữ lại từ ghép
    text = re.sub(r'[^\w\s\.\,\!\?\:\;\/\(\)_]', ' ', text)
    
    # --- Dọn dẹp khoảng trắng và dấu câu lặp lại ---
    text = re.sub(r'([.?!,])\1+', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def process_data(split_dataset):
    """Hàm xử lý dữ liệu"""
    df = pd.DataFrame(split_dataset)
    # Xử lý văn bản
    df['article'] = df['article'].progress_apply(clean_text)
    df['abstract'] = df['abstract'].progress_apply(clean_text)

    # Tính độ dài văn bản
    df['article_len'] = df['article'].apply(lambda x: len(x.split()))
    df['abstract_len'] = df['abstract'].apply(lambda x: len(x.split()))

    # Lọc dữ liệu
    df.dropna(inplace=True, subset=['article', 'abstract'])
    df.drop_duplicates(inplace=True, subset=['article', 'abstract'], keep='first')
    df = df[df['article_len'] >= MIN_ARTICLE_LEN]
    df = df[df['abstract_len'] >= MIN_SUMMARY_LEN]
    
    return df

def save_dataset(df, split_name):
    """Hàm lưu dữ liệu đã xử lý vào file csv"""
    output_path = os.path.join(PROCESSED_DATA_DIR, f'{split_name}.csv')
    df.to_csv(output_path, index=False)
    print(f'Đã lưu dữ liệu đã xử lý vào {output_path}')

    # Lưu mẫu dữ liệu để kiểm tra
    sample_path = os.path.join(PROCESSED_DATA_DIR, f'{split_name}_sample.txt')
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write("--- MẪU DỮ LIỆU SAU KHI XỬ LÝ ---\n\n")
        sample_df = df.sample(3)
        for i, row in sample_df.iterrows():
            f.write(f"--- Mẫu {i+1} ---\n")
            f.write(f"Article:\n{row['article']}\n\n")
            f.write(f"Abstract:\n{row['abstract']}\n\n")
