import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from preprocessing.dataset import SupportTicketDataset
from model.bert import BERTClassifier


def evaluate(model, loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased",
        local_files_only=True,
    )

    val_dataset = SupportTicketDataset("data/val.csv", tokenizer, max_length=128)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = BERTClassifier(num_labels=5).to(device)
    state = torch.load("model.pth", map_location=device)
    model.load_state_dict(state)

    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"Val loss: {val_loss:.4f} | Val accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
