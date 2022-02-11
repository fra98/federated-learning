from pydoc import cli
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from copy import deepcopy


def load_cifar(name, train, randomize=False):
    if name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10
        mean = (0.49139968, 0.48215841, 0.44653091)
        std = (0.24703223, 0.24348513, 0.26158784)
        image_size = 32
    elif name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100
        mean = (0.50707516, 0.48654887, 0.44091784)
        std = (0.26733429, 0.25643846, 0.27615047)
        image_size = 32
    else:
        raise NameError("Dataset not implemented")

    if randomize == True and train == False:
        raise NameError("Warning: Applying random transformations on testset")

    if train and randomize:
        transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10, expand=False),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    return dataset(root=f'./data/{name}', train=train, download=True, transform=transform)


def run_accuracy(device, dataset, batch_size, net, criterion): 
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    val_samples = len(dataset)
    val_steps = len(testloader)
    val_loss_epoch = 0
    num_correct = 0

    net.eval()
    with torch.no_grad():
        for data in testloader:
            # calculate outputs by running images through the network
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)

            # the class with the highest energy is what we choose as prediction
            val_loss = criterion(outputs, labels)
            val_loss_step = val_loss.data.item()
            val_loss_epoch += val_loss_step
            _, predicted = torch.max(outputs.data, 1)
            num_correct += torch.sum(predicted == labels.data).data.item()
        
        val_accuracy = (num_correct / val_samples) * 100
        avg_val_loss = val_loss_epoch / val_steps

    return val_accuracy, avg_val_loss


def generate_clients_sizes(dataset_size, num_clients, std_client_samples):
    avg_train_size = dataset_size / num_clients
    clients_sizes = np.random.normal(avg_train_size, avg_train_size * std_client_samples, num_clients)
    delta = dataset_size - np.sum(clients_sizes) # distribute difference over all clients
    clients_sizes = np.around((clients_sizes + delta/len(clients_sizes))).astype(int)  
    clients_sizes = max(clients_sizes, int(avg_train_size * 0.1))
    clients_sizes = min(clients_sizes, int(avg_train_size * 1.9))
    return clients_sizes


def get_class_priors(num_classes, labels, device='cpu'):
    labels = torch.tensor(labels, dtype=torch.int).to(device)
    priors = torch.zeros(num_classes)
    for i in range(num_classes):
        priors[i] = torch.sum(i == labels).data.item()

    return priors / torch.sum(priors)

def get_optimizer(conf_optimizer, train_params):
    optimizer = None
    if conf_optimizer["name"] == "Adam":
        optimizer = torch.optim.Adam(train_params,
            lr=conf_optimizer["lr"],
            weight_decay=conf_optimizer["weight_decay"],
            betas=[conf_optimizer["beta1"], conf_optimizer["beta2"]])
        if conf_optimizer["tau"] is not None:
            optimizer.eps = conf_optimizer["tau"]
    elif conf_optimizer["name"] == "Adagrad":
        optimizer = torch.optim.Adagrad(train_params,
            lr=conf_optimizer["lr"],
            weight_decay=conf_optimizer["weight_decay"])
        if conf_optimizer["tau"] is not None:
            optimizer.eps = conf_optimizer["tau"]
    elif conf_optimizer["name"] == "SGD":
        optimizer = torch.optim.SGD(train_params,
            lr=conf_optimizer["lr"],
            momentum=conf_optimizer["momentum"],
            weight_decay=conf_optimizer["weight_decay"])
    return optimizer