import torch
import torchvision
import torchvision.transforms as transforms


def load_cifar(name, train):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if name == 'CIFAR10':
        return torchvision.datasets.CIFAR10(root='./data/cifar10', train=train, download=True, transform=transform)
    elif name == 'CIFAR100':
        return torchvision.datasets.CIFAR100(root='./data/cifar100', train=train, download=True, transform=transform)
    else:
        raise NameError("Dataset not found")