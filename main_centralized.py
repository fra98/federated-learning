import torch
import torch.nn as nn
import torch.optim as optim

from src.models import *
from src.utils import load_cifar, run_accuracy


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #'cuda' or 'cpu' 

DATASET_NAME = "CIFAR10" 
BATCH_SIZE = 64 
NUM_EPOCHS = 5           # Total number of training epochs (iterations over dataset)

LR = 0.01                 # The initial Learning Rate
MOMENTUM = 0.9            # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 4e-4       # Regularization, you can keep this at the default
STEP_DOWN = False
STEP_SIZE = [5, 10, 15]   # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 0.1               # Multiplicative factor for learning rate step-down

NET = LeNet5_MOD_LIGHT_BN


def run_training(net, criterion, optimizer, scheduler=None):
    DROP_LAST = False
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=DROP_LAST)
    iter_per_epoch = len(trainloader)
    if DROP_LAST:
        train_samples = len(trainset) - (len(trainset) % BATCH_SIZE)
    else:
        train_samples = len(trainset)

    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        epoch_loss = 0
        num_corr_train = 0

        for _, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #compute the training accuracy 
            _, predicted = torch.max(outputs.data, 1)
            num_corr_train += torch.sum(predicted == labels.data).data.item()
            step_loss = loss.data.item()
            epoch_loss += step_loss

        avg_loss = epoch_loss / iter_per_epoch
        train_accuracy = (num_corr_train / train_samples) * 100
        print(f'Train: Epoch = {epoch+1} | Loss = {avg_loss:.3f} | Accuracy = {train_accuracy:.3f}')
        
        if scheduler is not None:
            scheduler.step()   

if __name__ == '__main__':
    print(DEVICE)

    trainset = load_cifar(name=DATASET_NAME, train=True)
    testset = load_cifar(name=DATASET_NAME, train=False)
    num_classes = len(trainset.classes)

    net = NET(num_classes)
    net.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.SGD(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    scheduler = None
    if STEP_DOWN:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=STEP_SIZE, gamma=GAMMA)
    
    run_training(net, criterion, optimizer, scheduler)

    accuracy, loss = run_accuracy(DEVICE, testset, BATCH_SIZE, net, criterion)
    print(f'Test Set: Loss {loss:.3f} | Accuracy = {accuracy:.3f}')

