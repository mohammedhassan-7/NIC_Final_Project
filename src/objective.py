import torch
import torch.nn as nn
import torch.optim as optim
from .model import TinyBERTClassifier

def obj_function(params, train_loader_subset, val_loader, num_classes, device):
    # 1. Unpack parameters
    lr = params[0]
    dropout = params[1]
    batch_size = int(params[2])

    # 2. Initialize TinyBERT Model
    model = TinyBERTClassifier(num_classes, dropout=dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. Train for 4 Epochs on the SUBSET
    model.train()
    for epoch in range(4):
        for batch in train_loader_subset:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # 4. Evaluate validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()

    accuracy = correct / total
    # Return accuracy to maximize
    return accuracy