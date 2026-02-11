import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

URGENT_KEYWORDS = [
    "blocked",
    "breach",
    "breached",
    "chargeback",
    "compromised",
    "critical",
    "data loss",
    "ddos",
    "deducted",
    "down",
    "escalate",
    "escalation",
    "fraud",
    "fraudulent",
    "hack",
    "incident",
    "lockout",
    "locked",
    "outage",
    "p0",
    "p1",
    "payment failed",
    "priority",
    "ransomware",
    "refund",
    "security",
    "service unavailable",
    "sev1",
    "severe",
    "suspended",
    "system down",
    "unable",
    "urgent",
    "overcharged",
    "double charged",
    "error 500",
]

CRITICAL_LABELS = {1, 2, 0}


def classify_urgency(text, predicted_label):
    text_lower = text.lower()
    if any(word in text_lower for word in URGENT_KEYWORDS):
        return "High"
    if predicted_label in CRITICAL_LABELS:
        return "Medium"
    return "Low"


def urgency_from_logits(text, logits):
    probs = F.softmax(logits, dim=-1)
    predicted_label = probs.argmax(dim=-1).item()
    return classify_urgency(text, predicted_label)


class BERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",
            local_files_only=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.classifier(x)
