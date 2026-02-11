# support_ticket_ai

A simple support-ticket classifier that predicts the ticket category and urgency using a fine-tuned BERT model. It includes:
- A FastAPI JSON API
- A small HTML UI for manual testing

## What the app does
- Takes a support ticket text input
- Predicts a category label (Billing, Technical, Account, Feature, Complaint)
- Computes a simple urgency signal (High/Medium/Low)

## Requirements
- Python 3.10+ (tested with 3.12)
- `pip`

## Setup
1. Create and activate a virtual environment.
2. Install dependencies.

Example (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you use the HTML form (`/predict`), FastAPI requires:
```powershell
pip install python-multipart
```

## Model files
The trained weights are **not** stored in the repo (GitHub file size limits). The app expects a file called `model.pth` in the project root:
```
model.pth
```

To create it, run training:
```powershell
python training\train.py
```

Notes:
- The code uses `local_files_only=True` when loading `bert-base-uncased`. Make sure the model/tokenizer are already cached locally, or download them once before running offline.

## Run the app
Start the server:
```powershell
uvicorn app:app --reload
```

Then open:
- UI: http://127.0.0.1:8000
- API docs: http://127.0.0.1:8000/docs

## API usage
JSON endpoint:
```http
POST /api/predict
Content-Type: application/json

{ "text": "My account is locked" }
```

Response:
```json
{
  "label_id": 2,
  "label": "Account",
  "confidence": 0.92,
  "urgency": "High",
  "scores": [0.01, 0.02, 0.92, 0.03, 0.02]
}
```

HTML form endpoint:
- `POST /predict` (used by the UI form at `/`)

## Config
You can adjust settings in `config.py`:
- Model name
- Number of labels
- Max length
- Device selection (`cpu` or `cuda`)
