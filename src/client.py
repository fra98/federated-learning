import torch
import torch.nn as nn
import torch.optim as optim

from .models import *
from .utils import run_accuracy


class Client:
    def __init__(self, id, device, local_epochs, batch_size, trainset, model_config, optim_config,
                 class_probabilities=None):
        self.id = id
        self.device = device
        self.local_epochs = local_epochs

        # DATASET CONFIGURATION
        self.batch_size = batch_size
        self.trainset = trainset
        self.trainset_size = len(trainset)
        self.num_classes = len(self.trainset.dataset.classes)  # trainset is a subset! access .dataset to obtain classes
        self.class_probabilities = class_probabilities

        # MODEL CONFIGURATION
        self.model_config = model_config
        self.optim_config = optim_config
        self.net = eval(self.model_config["net"])(self.num_classes)

    def client_update(self, state_dict, drop_last=False):
        # Init net with current weights
        self.net.to(self.device)
        self.net.load_state_dict(state_dict)

        # Criterion and optimizer
        criterion = eval(self.model_config["criterion"])()
        trainable_params = [p for p in self.net.parameters() if p.requires_grad]
        optimizer = eval(self.model_config["optimizer"])(trainable_params, lr=self.optim_config["lr"],
                                                         momentum=self.optim_config["momentum"],
                                                         weight_decay=self.optim_config["weight_decay"])

        # Trainloader
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True,
                                                  num_workers=2, drop_last=drop_last)

        if drop_last:
            train_samples = self.trainset_size - (self.trainset_size % self.batch_size)
        else:
            train_samples = self.trainset_size
        iter_per_epoch = len(trainloader)

        for epoch in range(self.local_epochs):  # loop over the dataset multiple times
            epoch_loss = 0
            num_corr_train = 0

            for _, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                # FedIR implementation
                if self.class_probabilities is not None:
                    probabilities = torch.zeros((self.num_classes)).to(self.device)
                    for i in range(self.num_classes):
                        probabilities[i] = torch.sum(i == labels.data).data.item()
                    probabilities = probabilities/torch.sum(probabilities)
                    weight = (self.class_probabilities/probabilities).detach()
                    #print(weight)
                    criterion = eval(self.model_config["criterion"])(weight=weight)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # compute the training accuracy
                _, predicted = torch.max(outputs.data, 1)
                num_corr_train += torch.sum(predicted == labels.data).data.item()
                step_loss = loss.data.item()
                epoch_loss += step_loss

            avg_loss = epoch_loss / iter_per_epoch
            train_accuracy = (num_corr_train / train_samples) * 100
            print(f'Client {self.id} -> Train: Epoch = {epoch+1} | Loss = {avg_loss:.3f} | Accuracy = {train_accuracy:.3f}')

    def train_accuracy(self, state_dict):
        self.net.to(self.device)
        if state_dict is not None:
            self.net.load_state_dict(state_dict)
        criterion = eval(self.model_config["criterion"])()

        return run_accuracy(device=self.device, dataset=self.trainset, batch_size=self.batch_size, net=self.net,
                            criterion=criterion)
