import torch
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request
from transformers import BertTokenizer

import config
from model.bert import BERTClassifier, urgency_from_logits

app = FastAPI(title="Support Ticket AI")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

LABELS = ["Billing", "Technical", "Account", "Feature", "Complaint"]


def _get_device() -> torch.device:
    preferred = getattr(config, "DEVICE", "auto")
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred.startswith("cuda"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


device = _get_device()

tokenizer = BertTokenizer.from_pretrained(
    getattr(config, "MODEL_NAME", "bert-base-uncased"),
    local_files_only=True,
)

model = BERTClassifier(num_labels=getattr(config, "NUM_LABELS", 5)).to(device)
state = torch.load("model.pth", map_location=device)
model.load_state_dict(state)
model.eval()


class PredictRequest(BaseModel):
    text: str


def _predict(text: str):
    encoded = tokenizer(
        text,
        max_length=getattr(config, "MAX_LENGTH", 128),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)[0]

    probs = torch.softmax(logits, dim=-1)
    label_id = int(probs.argmax().item())
    confidence = float(probs[label_id].item())
    label = LABELS[label_id] if label_id < len(LABELS) else f"Label {label_id}"
    urgency = urgency_from_logits(text, logits)

    return {
        "label_id": label_id,
        "label": label,
        "confidence": confidence,
        "urgency": urgency,
        "scores": [float(p) for p in probs.tolist()],
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None, "text": ""},
    )


@app.post("/predict", response_class=HTMLResponse)
def predict_form(request: Request, text: str = Form("")):
    text = text.strip()
    if not text:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": None, "text": ""},
        )

    result = _predict(text)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "text": text},
    )


@app.post("/api/predict")
def predict_api(payload: PredictRequest):
    text = payload.text.strip()
    if not text:
        return {"error": "text is required"}
    return _predict(text)
