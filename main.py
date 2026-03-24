import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data_loader import Data_Loader
from model import PetNet
from train import train_model, validate_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

experiments = [
    {"name": "Basic", "bn": False, "dropout": 0.0, "aug": False, "lr": 0.001},
    {"name": "Regularization", "bn": True, "dropout": 0.3, "aug": False, "lr": 0.001},
    {"name": "Maxed Out", "bn": True, "dropout": 0.5, "aug": True, "lr": 0.0005},
]

results = {}

for exp in experiments:
    print(f"\n--- Experiment {exp['name']} ---")
    train_loader, test_loader, _ = Data_Loader(use_augmentation=exp['aug'])
    model = PetNet(use_batchnorm=exp['bn'], dropout_p=exp['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp['lr'])
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, 11):
        loss, acc = train_model(model, train_loader, criterion, optimizer, device)
        test_acc = validate_model(model, test_loader, device)
        history.append(test_acc)
        print(f"\n--- Epoch {epoch}: Train Accuracy: {acc:.4f}% -- Test Accuracy: {test_acc:.4f}% -- Loss: {loss:.4f} ---")
    results[exp['name']] = history

plt.figure(figsize=(10, 6))
for name, hist in results.items():
    plt.plot(hist, label=name)
plt.title("Comparing Experiments - Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()