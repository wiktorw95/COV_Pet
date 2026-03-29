import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

def Data_Loader(batch_size=32, use_augmentation=False):
    transform = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]

    train_transform = transform.copy()
    if use_augmentation:
        train_transform.insert(0, transforms.RandomHorizontalFlip())
        train_transform.insert(0, transforms.RandomRotation(15))
        train_transform.insert(0, transforms.ColorJitter(brightness=0.2, contrast=0.2))
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(transform)

    train_set = torchvision.datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=train_transform)
    test_set = torchvision.datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=test_transform)

    train_loadered = DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True)
    test_loadered = DataLoader(test_set, batch_size=batch_size, num_workers=2, shuffle=False)

    return train_loadered, test_loadered, train_set.classes