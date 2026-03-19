import torch
from tqdm import tqdm

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    pbar = tqdm(train_loader, desc='Training', unit='batch')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pbar.set_postfix(loss=loss.item())

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=loss.item(), acc=100. * correct / ((pbar.n + 1) * train_loader.batch_size))
    return total_loss / len(train_loader), 100. * correct / len(train_loader.dataset)

def validate_model(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / len(test_loader.dataset)