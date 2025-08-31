import torch
import torch.nn.functional as F
import argparse
from pathlib import Path

from utils.process_data import clean_text, post_process_summary
from utils.model_loader import load_model_resources

def _truncate_tokens(tokens, max_len: int, strategy: str):
    """
    Hàm helper để cắt chuỗi token theo chiến lược được chọn.
    - head: lấy phần đầu
    - tail: lấy phần cuối
    - head_tail: lấy nửa đầu và nửa cuối
    """
    if max_len <= 0:
        return []
    strategy = (strategy or "head").lower()
    if strategy == "tail":
        return tokens[-max_len:]
    if strategy == "head_tail":
        front_len = max_len // 2 + (max_len % 2)
        back_len = max_len - front_len
        if back_len <= 0:
            return tokens[:front_len]
        return tokens[:front_len] + tokens[-back_len:]
    return tokens[:max_len]

def summarize_with_beam_search(text_to_summarize: str, model, tokenizer, config, device, beam_width=5, temperature=1.0):
    """Hàm trả về tóm tắt sử dụng beam search."""
    model.eval()
    
    source_tokens = tokenizer.encode_as_ids(text_to_summarize)
    
    max_len_for_tokens = config['max_len_text'] - 2
    if len(source_tokens) > max_len_for_tokens:
        strategy = config.get('truncate_strategy_text', 'head')
        source_tokens = _truncate_tokens(source_tokens, max_len_for_tokens, strategy)

    source_tokens = [tokenizer.bos_id()] + source_tokens + [tokenizer.eos_id()]
    
    encoder_input = torch.tensor([source_tokens], dtype=torch.long, device=device)
    encoder_mask = (encoder_input != tokenizer.pad_id()).unsqueeze(1).unsqueeze(1).to(device)
    
    with torch.no_grad():
        model_out = beam_search_decode(
            model, encoder_input, encoder_mask, tokenizer, 
            config['max_len_summary'], device, beam_width, temperature
        )

    summary_text = tokenizer.decode(model_out.cpu().numpy().tolist())
    return summary_text

def beam_search_decode(model, source, source_mask, tokenizer, max_len, device, beam_width=5, temperature=1.0):
    """
    Sinh văn bản tóm tắt bằng thuật toán Beam Search.

    Args:
        model (torch.nn.Module): Mô hình Transformer đã được huấn luyện.
        source (torch.Tensor): Tensor chứa token ids của văn bản đầu vào.
        source_mask (torch.Tensor): Mask cho source sequence.
        tokenizer: Tokenizer tương ứng với mô hình.
        max_len (int): Độ dài tối đa của tóm tắt được sinh ra.
        device: Device để chạy model.
        beam_width (int): Số lượng beam cần duy trì.
        temperature (float): Yếu tố làm "mượt" phân phối xác suất.

    Returns:
        torch.Tensor: Chuỗi token ids của tóm tắt tốt nhất.
    """
    model.eval()
    
    sos_idx = tokenizer.bos_id()
    eos_idx = tokenizer.eos_id()
    
    encoder_output = model.encode(source, source_mask)
    
    # Khởi tạo beams với SOS token: [(decoder_input, log_prob)]
    beams = [(torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device), 0.0)]
    completed_beams = []
    
    for step in range(max_len - 1):
        if not beams:
            break
            
        all_candidates = []
        
        for decoder_input, score in beams:
            # Nếu beam đã kết thúc với EOS, chuyển vào completed_beams
            if decoder_input[0, -1].item() == eos_idx:
                completed_beams.append((decoder_input, score))
                continue
                
            # Tạo decoder mask
            seq_len = decoder_input.size(1)
            decoder_mask = torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1).type(torch.bool).to(device)
            
            # Forward pass qua decoder
            decoder_output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            
            # Lấy logits cho token cuối cùng
            logits = model.project(decoder_output[:, -1, :])
            
            # Áp dụng temperature và tính log probabilities
            if temperature != 1.0:
                logits = logits / temperature
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Lấy top-k tokens
            top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)
            
            # Tạo candidates mới
            for i in range(beam_width):
                next_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)
                token_log_prob = top_log_probs[0, i].item()
                
                # Tạo sequence mới
                new_decoder_input = torch.cat([decoder_input, next_token], dim=1)
                new_score = score + token_log_prob
                
                all_candidates.append((new_decoder_input, new_score))
        
        # Sắp xếp và chọn top beam_width candidates
        all_candidates.sort(key=lambda x: x[1] / x[0].size(1), reverse=True)
        beams = all_candidates[:beam_width]
    
    # Thêm các beam còn lại vào completed_beams
    for decoder_input, score in beams:
        completed_beams.append((decoder_input, score))
    
    # Chọn beam tốt nhất (normalize theo độ dài)
    if completed_beams:
        best_beam = max(completed_beams, key=lambda x: x[1] / x[0].size(1))
        return best_beam[0].squeeze(0)
    else:
        return beams[0][0].squeeze(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="=Tóm tắt văn bản tiếng Việt")
    parser.add_argument("--text", type=str, required=True, help="Văn bản cần tóm tắt")
    parser.add_argument("--beam_width", type=int, default=5, help="Số lượng beam cần duy trì")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature để làm 'mượt' phân phối xác suất")
    args = parser.parse_args()

    base_dir = str(Path(__file__).resolve().parent)
    
    print("--- Tải mô hình và tài nguyên... ---")
    try:
        model, tokenizer, config, device = load_model_resources(base_dir)
        print("--- Tải tài nguyên thành công. ---")

        print("\n--- Tóm tắt văn bản... ---")
        cleaned_text = clean_text(args.text)
        summary = summarize_with_beam_search(
            text_to_summarize=cleaned_text,
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            beam_width=args.beam_width,
            temperature=args.temperature,
        )
        final_summary = post_process_summary(summary)
        
        print("\n--- Kết quả tóm tắt ---")
        print(final_summary)
        print("---------------")

    except Exception as e:
        print(f"\n[Lỗi] Đã xảy ra lỗi: {e}")