from contextlib import asynccontextmanager
from pathlib import Path
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import sentencepiece as spm
from pydantic import BaseModel

from utils.config import get_config
from utils.process_data import clean_text, post_process_summary
from model.transformer import build_transformer
from inference import summarize_with_beam_search
from utils.model_loader import load_model_resources

class SummarizeRequest(BaseModel):
    text: str
    beam_width: int = 5
    temperature: float = 1.0

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
