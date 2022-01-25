import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from copy import deepcopy


def load_cifar(name, train):
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

    if train:
        transform = transforms.Compose([
            # transforms.RandomCrop(image_size, padding=4),
            # transforms.RandomHorizontalFlip(),
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
    net.train(False)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    val_samples = len(dataset)
    val_steps = len(testloader)
    val_loss_epoch = 0
    num_correct = 0

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


def indexes_split_IID(num_clients, trainset_size):
    # NOTE: this split assumes equal class distribution -> true for cifar
    indexes = []
    for i in range(num_clients):
        indexes.append(list(range(i, trainset_size, num_clients)))
    return indexes


def indexes_split_NON_IID(num_clients, num_classes, alpha, dataset):  # class-balanced
    # NOTE: this split assumes equal class distribution -> true for cifar
    client_data_size = len(dataset) // num_clients
    num_samples_per_class = len(dataset) // num_classes
    classes = [i for i in range(num_classes)]

    # Indexes sorted by class label 
    sorted_indexes = np.argsort(dataset.targets)

    # Priors probabilities of each class
    priors = (1/num_classes) * np.ones(num_classes)

    # Dirichlet's distribution probabilities -> 2D matrix, shape: (NUM_CLIENTS, NUM_CLASSES)
    dirichlet_probs = np.random.dirichlet(alpha * priors, num_clients)

    indexes = []
    for _ in range(num_clients):
        indexes.append([])

    offsets_class = np.zeros(num_classes, dtype=np.int32)
    shuffled_classes = deepcopy(classes)
    for user_id in range(num_clients):      
        for _ in range(client_data_size):     
            class_label = np.random.choice(classes, p=dirichlet_probs[user_id])

            # check if selected class is available. If not randomly select another one  
            if offsets_class[class_label] >= num_samples_per_class:
                random.shuffle(shuffled_classes)
                for c in shuffled_classes:
                    if offsets_class[c] < num_samples_per_class:
                        class_label = c

            pos = num_samples_per_class * class_label + offsets_class[class_label]  # base + offset
            image_id = sorted_indexes[pos]
            indexes[user_id].append(image_id)

            offsets_class[class_label] += 1

    return indexes


def get_class_priors(num_classes, labels, device='cpu'):
    labels = torch.tensor(labels, dtype=torch.int).to(device)
    priors = torch.zeros(num_classes)
    for i in range(num_classes):
        priors[i] = torch.sum(i == labels).data.item()

    return priors / torch.sum(priors)
