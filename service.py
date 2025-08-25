import torch
import sentencepiece as spm
from pathlib import Path
from typing import Optional
import logging

from model.transformer import build_transformer
from utils.config import get_config
from inference import summarize_with_beam_search

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SummarizerService:
    """Service để load model và thực hiện inference tóm tắt văn bản"""
    
    def __init__(self, config_base_path: str = "."):
        self.config = get_config(config_base_path)
        self.device = self.config['device']
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer: Optional[spm.SentencePieceProcessor] = None
        self._is_loaded = False
        
    def load_model_and_tokenizer(self):
        """Tải model và tokenizer từ checkpoint"""
        try:
            logger.info("Đang tải tokenizer...")
            self.tokenizer = spm.SentencePieceProcessor()
            tokenizer_path = self.config["tokenizer_file"]
            if not Path(tokenizer_path).exists():
                raise FileNotFoundError(f"Không tìm thấy tokenizer tại: {tokenizer_path}")
            self.tokenizer.load(tokenizer_path)
            logger.info("Tokenizer đã được tải thành công!")
            
            logger.info("Đang tải model...")
            self.model = build_transformer(
                src_vocab_size=self.tokenizer.get_piece_size(),
                tgt_vocab_size=self.tokenizer.get_piece_size(),
                src_seq_len=self.config["max_len_text"],
                tgt_seq_len=self.config["max_len_summary"],
                d_model=self.config["d_model"],
                N=self.config["num_layers"],
                h=self.config["num_heads"],
                d_ff=self.config["d_ff"],
                dropout=self.config["dropout"],
            ).to(self.device)
            
            # Tải checkpoint
            model_path = Path(self.config["save_dir"]) / "transformer_summarizer_best.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"Không tìm thấy model checkpoint tại: {model_path}")
                
            state = torch.load(model_path, map_location=self.device)
            
            # Xử lý state dict (loại bỏ prefix _orig_mod. nếu có)
            state_dict = state['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('_orig_mod.'):
                    new_state_dict[k[len('_orig_mod.'):]] = v
                else:
                    new_state_dict[k] = v
            
            self.model.load_state_dict(new_state_dict)
            self.model.eval()
            
            self._is_loaded = True
            logger.info("Model đã được tải thành công!")
            
        except Exception as e:
            logger.error(f"Lỗi khi tải model/tokenizer: {str(e)}")
            raise e
    
    def is_ready(self) -> bool:
        """Kiểm tra xem service đã sẵn sàng để inference chưa"""
        return self._is_loaded and self.model is not None and self.tokenizer is not None
    
    def summarize(self, text: str, beam_width: int = 5, temperature: float = 1.0) -> str:
        """
        Thực hiện tóm tắt văn bản
        
        Args:
            text: Văn bản cần tóm tắt
            beam_width: Số beam cho beam search (1-10)
            temperature: Temperature cho sampling (0.1-2.0)
        
        Returns:
            Văn bản tóm tắt
        """
        if not self.is_ready():
            raise RuntimeError("Service chưa được khởi tạo. Hãy gọi load_model_and_tokenizer() trước.")
        
        if not text or not text.strip():
            raise ValueError("Văn bản đầu vào không được rỗng")
        
        # Validate parameters
        beam_width = max(1, min(10, beam_width))
        temperature = max(0.1, min(2.0, temperature))
        
        try:
            summary = summarize_with_beam_search(
                text_to_summarize=text.strip(),
                model=self.model,
                tokenizer=self.tokenizer,
                config=self.config,
                device=self.device,
                beam_width=beam_width,
                temperature=temperature
            )
            return summary
        except Exception as e:
            logger.error(f"Lỗi khi thực hiện tóm tắt: {str(e)}")
            raise e

# Global instance
summarizer_service = SummarizerService()
