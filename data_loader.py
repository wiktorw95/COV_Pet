import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def Data_Loader(batch_size=32, use_augmentation=False):
    transform = [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    train_transform = transform.copy()
    if use_augmentation:
        train_transform.insert(0, transforms.RandomHorizontalFlip())
        train_transform.insert(0, transforms.RandomRotation(15))
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(transform)

    train_set = torchvision.datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=train_transform)
    test_set = torchvision.datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_set.classes