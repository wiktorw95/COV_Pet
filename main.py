import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data_loader import Data_Loader
from model import PetResNet, PetNet
from train import train_model, validate_model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 7

    experiments = [
        {"name": "Base_NoAug", "bn": False, "dropout": 0, "aug": False, "lr": 0.0005, "wd": 0},
        {"name": "Aug_LowDrop", "bn": True, "dropout": 0.3, "aug": False, "lr": 0.0005, "wd": 1e-4},
        {"name": "Aug_HighDrop", "bn": True, "dropout": 0.2, "aug": True, "lr": 0.001, "wd": 1e-4},
        {"name": "ResNet18", "bn": True, "dropout": 0.3, "aug": True, "lr": 0.0003, "wd": 1e-4}
    ]

    all_results = {}

    for exp in experiments:
        print(f"\n{'=' * 40}")
        print(f" Experiment: {exp['name']}")
        print(f"Parameters: BatchNorm={exp['bn']}, Dropout={exp['dropout']}, Augmentation={exp['aug']}, LR={exp['lr']}")
        print(f"{'=' * 40}")

        train_loader, test_loader, classes = Data_Loader(use_augmentation=exp['aug'])

        if (exp['name'] == "ResNet18_Transfer"):
            model = PetResNet(num_classes=len(classes), dropout_p=exp['dropout']).to(device)
        else:
            model = PetNet(num_classes=len(classes), use_batchnorm=exp["bn"], dropout_p=exp['dropout']).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=exp['lr'], weight_decay=exp["wd"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        history_test = []
        history_train = []

        best_test_acc = 0.0
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
            test_acc = validate_model(model, test_loader, device)

            history_train.append(train_acc)
            history_test.append(test_acc)
            scheduler.step(test_acc)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                epochs_no_improve = 0
                marker = " ⭐ NEW BEST!"
            else:
                epochs_no_improve += 1
                marker = ""

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoka {epoch:02d} | LR: {current_lr:.6f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Najlepszy: {best_test_acc:.2f}%{marker}")

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(
                    f"🛑 Early stopping triggered!")
                break

        all_results[exp['name']] = {
            'test': history_test,
            'train': history_train,
            'best_acc': best_test_acc,
            'best_epoch': best_epoch
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Podsumowanie Eksperymentów: Train vs Test Accuracy", fontsize=16, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Matplotlib defaults for clarity

    for idx, (name, data) in enumerate(all_results.items()):
        color = colors[idx % len(colors)]
        epochs_range = range(1, len(data['test']) + 1)

        # Left Panel: Train Accuracy (Dashed lines)
        ax1.plot(epochs_range, data['train'], label=f"{name}", color=color, linestyle='--')

        # Right Panel: Test Accuracy (Solid lines)
        ax2.plot(epochs_range, data['test'], label=f"{name} (Max: {data['best_acc']:.1f}%)", color=color, linewidth=2)

        # Place a Star marker exactly where the model saved its best weights
        ax2.scatter(data['best_epoch'], data['best_acc'], color=color, s=150, zorder=5, edgecolor='black', marker='*')

        # Add a tiny text label next to the star
        ax2.annotate(f"{data['best_acc']:.1f}%",
                     (data['best_epoch'], data['best_acc']),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center', fontsize=10, color=color, fontweight='bold')

    # Formatting Left Graph
    ax1.set_title("Train Accuracy (Zdolność do zapamiętywania)")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Dokładność (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Formatting Right Graph
    ax2.set_title("Test Accuracy (Zdolność do generalizacji)")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Dokładność (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()