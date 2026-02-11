import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from preprocessing.dataset import SupportTicketDataset
from model.bert import BERTClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# offline-friendly: require local cache of the model/tokenizer
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased",
    local_files_only=True,
)

train_dataset = SupportTicketDataset("data/train.csv", tokenizer)
val_dataset = SupportTicketDataset("data/val.csv", tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

model = BERTClassifier(num_labels=5).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

torch.save(model.state_dict(), "model.pth")
print("Model saved.")
