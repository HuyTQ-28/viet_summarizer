import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from .dataset import VietNewSumDataset


def collate_fn(batch):
    """
    Collate function cho DataLoader để xử lý batch data với dynamic padding.
    Đệm các chuỗi đến độ dài lớn nhất trong batch và tạo các attention mask tương ứng.
    """
    if not batch:
        raise ValueError("Batch không thể rỗng!")
    
    # Lấy pad_id
    pad_id = batch[0]['pad_id']

    # Tách các thành phần từ batch
    encoder_inputs = [item['encoder_input'] for item in batch]
    decoder_inputs = [item['decoder_input'] for item in batch]
    labels = [item['label'] for item in batch]
    src_texts = [item['src_text'] for item in batch]
    tgt_texts = [item['tgt_text'] for item in batch]

    # Sử dụng pad_sequence để đệm các chuỗi trong batch
    encoder_inputs_padded = pad_sequence(encoder_inputs, batch_first=True, padding_value=pad_id)
    decoder_inputs_padded = pad_sequence(decoder_inputs, batch_first=True, padding_value=pad_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=pad_id)

    # Tạo encoder_mask: (batch_size, 1, 1, src_seq_len)
    # Mask này dùng để che đi các token padding trong encoder input.
    encoder_mask = (encoder_inputs_padded != pad_id).unsqueeze(1).unsqueeze(1)

    # Tạo decoder_mask (kết hợp padding mask và causal mask)
    # 1. Tạo padding mask cho decoder: (batch_size, 1, tgt_seq_len)
    decoder_padding_mask = (decoder_inputs_padded != pad_id).unsqueeze(1)
    
    # 2. Tạo causal mask: (tgt_seq_len, tgt_seq_len)
    tgt_seq_len = decoder_inputs_padded.size(1)
    # Chuyển causal mask đến cùng device với data
    device = decoder_inputs_padded.device
    causal_mask = create_causal_mask(tgt_seq_len).to(device)

    # 3. Kết hợp 2 mask: (batch_size, tgt_seq_len, tgt_seq_len)
    # Causal mask sẽ được broadcast để khớp với shape của decoder_padding_mask
    decoder_mask = decoder_padding_mask.unsqueeze(1) & causal_mask.unsqueeze(0)

    return {
        'encoder_input': encoder_inputs_padded,
        'decoder_input': decoder_inputs_padded,
        'label': labels_padded,
        'encoder_mask': encoder_mask,
        'decoder_mask': decoder_mask,
        'src_text': src_texts,
        'tgt_text': tgt_texts
    }


# Cache cho causal masks để tránh tạo lại nhiều lần
_causal_mask_cache = {}

def create_causal_mask(size):
    """
    Tạo causal mask cho decoder (lower triangular matrix) với caching.
    Args:
        size (int): Kích thước của sequence
    Returns:
        torch.Tensor: Causal mask có shape (size, size)
    """
    if size not in _causal_mask_cache:
        mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
        _causal_mask_cache[size] = ~mask  # Invert để True là vị trí được phép nhìn thấy
    return _causal_mask_cache[size]


def clear_causal_mask_cache():
    """
    Xóa cache của causal masks (hữu ích khi muốn giải phóng bộ nhớ).
    """
    global _causal_mask_cache
    _causal_mask_cache.clear()


def create_data_loader(dataframe, tokenizer, max_len_text, max_len_summary,
                      batch_size, shuffle=True, num_workers=0,
                      truncate_strategy_text: str = "head",
                      truncate_strategy_summary: str = "head"):
    """
    Tạo DataLoader cho dataset.
    
    Args:
        dataframe (pd.DataFrame): DataFrame chứa dữ liệu
        tokenizer: SentencePiece tokenizer
        max_len_text (int): Chiều dài tối đa của văn bản nguồn
        max_len_summary (int): Chiều dài tối đa của tóm tắt
        batch_size (int): Kích thước batch
        shuffle (bool): Có shuffle dữ liệu không
        num_workers (int): Số worker cho DataLoader
        
    Returns:
        DataLoader: DataLoader đã được cấu hình
    """
    dataset = VietNewSumDataset(
        dataframe=dataframe,
        tokenizer=tokenizer,
        max_len_text=max_len_text,
        max_len_summary=max_len_summary,
        truncate_strategy_text=truncate_strategy_text,
        truncate_strategy_summary=truncate_strategy_summary,
    )
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def load_data_from_csv(train_path, val_path=None, test_path=None):
    """
    Load dữ liệu từ file CSV.
    
    Args:
        train_path (str): Đường dẫn đến file train CSV
        val_path (str): Đường dẫn đến file validation CSV (tùy chọn)
        test_path (str): Đường dẫn đến file test CSV (tùy chọn)
        
    Returns:
        tuple: (train_df, val_df, test_df) - val_df và test_df có thể là None
    """
    train_df = pd.read_csv(train_path)
    
    val_df = None
    if val_path:
        val_df = pd.read_csv(val_path)
    
    test_df = None 
    if test_path:
        test_df = pd.read_csv(test_path)
        
    return train_df, val_df, test_df


def get_data_loaders(train_df, tokenizer, config, val_df=None, test_df=None):
    """
    Tạo tất cả DataLoader cần thiết cho training.
    
    Args:
        train_df (pd.DataFrame): DataFrame training
        tokenizer: SentencePiece tokenizer  
        config (dict): Dictionary chứa cấu hình
        val_df (pd.DataFrame): DataFrame validation (tùy chọn)
        test_df (pd.DataFrame): DataFrame test (tùy chọn)
        
    Returns:
        dict: Dictionary chứa các DataLoader
    """
    data_loaders = {}
    
    # Training DataLoader
    data_loaders['train'] = create_data_loader(
        dataframe=train_df,
        tokenizer=tokenizer,
        max_len_text=config['max_len_text'],
        max_len_summary=config['max_len_summary'],
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        truncate_strategy_text=config.get('truncate_strategy_text', 'head'),
        truncate_strategy_summary=config.get('truncate_strategy_summary', 'head'),
    )
    
    # Validation DataLoader
    if val_df is not None:
        data_loaders['val'] = create_data_loader(
            dataframe=val_df,
            tokenizer=tokenizer,
            max_len_text=config['max_len_text'],
            max_len_summary=config['max_len_summary'],
            batch_size=config.get('val_batch_size', 1),
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            truncate_strategy_text=config.get('truncate_strategy_text', 'head'),
            truncate_strategy_summary=config.get('truncate_strategy_summary', 'head'),
        )
    
    # Test DataLoader
    if test_df is not None:
        data_loaders['test'] = create_data_loader(
            dataframe=test_df,
            tokenizer=tokenizer,
            max_len_text=config['max_len_text'],
            max_len_summary=config['max_len_summary'],
            batch_size=1,
            shuffle=False,
            num_workers=config.get('num_workers', 0),
            truncate_strategy_text=config.get('truncate_strategy_text', 'head'),
            truncate_strategy_summary=config.get('truncate_strategy_summary', 'head'),
        )
    
    return data_loaders