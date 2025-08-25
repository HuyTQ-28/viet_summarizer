from contextlib import asynccontextmanager
from pathlib import Path
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import sentencepiece as spm
from pydantic import BaseModel

from utils.config import get_config
from utils.process_data import clean_text, post_process_summary
from model.transformer import build_transformer
from inference import summarize_with_beam_search

class SummarizeRequest(BaseModel):
    text: str
    beam_width: int = 5
    temperature: float = 1.0

def load_model_resources(base_dir: str):
    """
    Tải tokenizer và model
    """
    config = get_config(base_dir)
    device = config['device']
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(config["tokenizer_file"])

    model = build_transformer(
        src_vocab_size=tokenizer.get_piece_size(),
        tgt_vocab_size=tokenizer.get_piece_size(),
        src_seq_len=config["max_len_text"],
        tgt_seq_len=config["max_len_summary"],
        d_model=config["d_model"],
        N=config["num_layers"],
        h=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(device)
    
    model_path = Path(config["save_dir"]) / "transformer_summarizer_best.pt"
    state = torch.load(model_path, map_location=device)
    
    state_dict = state['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[len('_orig_mod.'):]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)

    model.eval()
    return model, tokenizer, config, device

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời ứng dụng. Model và tokenizer được tải trên startup và xóa trên shutdown.
    """
    print("--- Server starting up: Loading resources ---")
    base_dir = str(Path(__file__).resolve().parent)
    try:
        model, tokenizer, config, device = load_model_resources(base_dir)
        app.state.model = model
        app.state.tokenizer = tokenizer
        app.state.config = config
        app.state.device = device
        print("--- Resources loaded successfully ---")
    except Exception as e:
        print(f"FATAL: Failed to load resources during startup: {e}")
        app.state.model = None
    
    yield 

    print("--- Server shutting down: Clearing resources ---")
    app.state.model = None
    app.state.tokenizer = None
    app.state.config = None
    app.state.device = None


BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Vietnamese Text Summarizer", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """Hiển thị trang chủ"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/summarize", response_class=JSONResponse)
async def summarize_text(
    request: Request,
    payload: SummarizeRequest
):
    """Xử lý yêu cầu tóm tắt văn bản"""
    if not request.app.state.model:
        raise HTTPException(
            status_code=503, detail="Model is not available due to a startup error."
        )

    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        processed_text = clean_text(payload.text)
        summary = summarize_with_beam_search(
            text_to_summarize=processed_text,
            model=request.app.state.model,
            tokenizer=request.app.state.tokenizer,
            config=request.app.state.config,
            device=request.app.state.device,
            beam_width=payload.beam_width,
            temperature=payload.temperature,
        )
        final_summary = post_process_summary(summary)
        return {"success": True, "summary": final_summary}
    except Exception as e:
        print(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
