import torch
from torch.utils.data import Dataset

class VietNewSumDataset(Dataset):
    """
    Lớp Dataset cho bài toán tóm tắt văn bản tiếng Việt.
    Nhận đầu vào là DataFrame, xử lý và chuyển đổi thành tensor cho mô hình Transformer.
    """

    def __init__(self, dataframe, tokenizer, max_len_text, max_len_summary,
                 truncate_strategy_text: str = "head",
                 truncate_strategy_summary: str = "head"):
        super().__init__()

        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len_text = max_len_text
        self.max_len_summary = max_len_summary
        self.truncate_strategy_text = truncate_strategy_text
        self.truncate_strategy_summary = truncate_strategy_summary

        # Lấy ID của các token đặc biệt từ tokenizer
        self.bos_id = self.tokenizer.bos_id()
        self.eos_id = self.tokenizer.eos_id()
        self.pad_id = self.tokenizer.pad_id()
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Lấy 1 hàng dữ liệu từ DataFrame
        row = self.dataframe.iloc[idx]  
        src_text = str(row['article'])
        tgt_text = str(row['abstract'])

        # Tokenize văn bản nguồn và đích
        src_tokens = self.tokenizer.encode_as_ids(src_text)
        tgt_tokens = self.tokenizer.encode_as_ids(tgt_text)

        # Cắt bớt token nếu quá dài (để dành chỗ cho BOS/EOS)
        if len(src_tokens) > self.max_len_text - 2:
            src_tokens = self._truncate_tokens(
                src_tokens,
                max_len=self.max_len_text - 2,
                strategy=self.truncate_strategy_text,
            )
        if len(tgt_tokens) > self.max_len_summary - 2:
            tgt_tokens = self._truncate_tokens(
                tgt_tokens,
                max_len=self.max_len_summary - 2,
                strategy=self.truncate_strategy_summary,
            )

        # Chuẩn bị encoder_input, decoder_input và label (chưa padding)
        encoder_input = [self.bos_id] + src_tokens + [self.eos_id]
        decoder_input = [self.bos_id] + tgt_tokens
        label = tgt_tokens + [self.eos_id]

        return {
            "encoder_input": torch.tensor(encoder_input, dtype=torch.long),
            "decoder_input": torch.tensor(decoder_input, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text,
            "pad_id": self.pad_id
        }

    def _truncate_tokens(self, tokens, max_len: int, strategy: str):
        """
        Cắt chuỗi token theo chiến lược được chọn.
        - head: lấy phần đầu
        - tail: lấy phần cuối
        - head_tail: lấy nửa đầu và nửa cuối (giữ cả bối cảnh mở và kết)
        """
        if max_len <= 0:
            return []
        strategy = (strategy or "head").lower()
        if strategy == "tail":
            return tokens[-max_len:]
        if strategy == "head_tail":
            # Chia gần đều cho 2 phía, ưu tiên đầu nếu lẻ
            front_len = max_len // 2 + (max_len % 2)
            back_len = max_len - front_len
            if back_len <= 0:
                return tokens[:front_len]
            return tokens[:front_len] + tokens[-back_len:]
        # Mặc định: head
        return tokens[:max_len]