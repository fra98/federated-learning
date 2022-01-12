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


def run_accuracy(device, dataset, batch_size, net, criterion) -> tuple[float, float]: 
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
