import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy

from .models import *
from .utils import run_accuracy, get_class_priors


class Client:
    def __init__(self, id, device, local_epochs, batch_size, trainset, model_config, optim_config,
                 server_class_priors=None, virtual_client_size=None, logger=None):
        self.id = id
        self.device = device
        self.local_epochs = local_epochs

        # DATASET CONFIGURATION
        self.batch_size = batch_size
        self.trainset = trainset
        self.trainset_size = len(trainset)

        # OPTMINIZATION CONFIGURATION
        self.num_classes = len(trainset.dataset.classes)  # trainset is a subset! access .dataset to obtain classes
        self.targets = [trainset.dataset.targets[idx] for idx in trainset.indices]
        self.class_priors = get_class_priors(self.num_classes, self.targets, self.device)
        self.server_class_priors = server_class_priors
        self.weight = server_class_priors / self.class_priors
        self.virtual_client_size = virtual_client_size
        self.logger = logger

        # MODEL CONFIGURATION
        self.model_config = model_config
        self.optim_config = optim_config
        self.net = eval(self.model_config["net"])(self.num_classes)
        self.starting_net = eval(self.model_config["net"])(self.num_classes)

    def client_update(self, state_dict, drop_last=False, fed_IR=False, print_acc=True, fed_VC=False):

        if fed_VC:
            trainset = fed_vc_get_random_subset(self.trainset, self.virtual_client_size)
        else:
            trainset = self.trainset

        trainset_size = len(trainset)

        # Init net with current weights
        self.net.to(self.device)
        self.net.load_state_dict(state_dict)
        self.net.train()
        self.net_updates = []


        # Criterion and optimizer
        if fed_IR:
            criterion = eval(self.model_config["criterion"])(weight=self.weight.to(self.device))
        else:
            criterion = eval(self.model_config["criterion"])()
        trainable_params = [p for p in self.net.parameters() if p.requires_grad]
        old_trainable_params = deepcopy(trainable_params)
        optimizer = eval(self.model_config["optimizer"])(trainable_params, lr=self.optim_config["lr"],
                                                         momentum=self.optim_config["momentum"],
                                                         weight_decay=self.optim_config["weight_decay"])

        # Trainloader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True,
                                                  num_workers=2, drop_last=drop_last)
        iter_per_epoch = len(trainloader)
        if drop_last:
            train_samples = trainset_size - (trainset_size % self.batch_size)
        else:
            train_samples = trainset_size

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

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # compute the training accuracy
                if print_acc:
                    _, predicted = torch.max(outputs.data, 1)
                    num_corr_train += torch.sum(predicted == labels.data).data.item()
                    step_loss = loss.data.item()
                    epoch_loss += step_loss

            if print_acc:
                avg_loss = epoch_loss / iter_per_epoch
                train_accuracy = (num_corr_train / train_samples) * 100
                self.logger.log(f'Client {self.id} -> Train: Epoch = {epoch + 1} | Loss = {avg_loss:.3f} | Accuracy = {train_accuracy:.3f}')

        for i in range(len(trainable_params)):
            self.net_updates[i] = trainable_params[i] - old_trainable_params[i]

    def train_accuracy(self, state_dict):
        self.net.to(self.device)
        if state_dict is not None:
            self.net.load_state_dict(state_dict)
        criterion = eval(self.model_config["criterion"])()

        return run_accuracy(device=self.device, dataset=self.trainset, batch_size=self.batch_size, net=self.net,
                            criterion=criterion)


def fed_vc_get_random_subset(set, virtual_client_size):
    set_size = len(set)
    replace = False
    if virtual_client_size > set_size:
        replace = True

    indexes = np.random.choice(set.indices, virtual_client_size, replace=replace)

    return torch.utils.data.Subset(set.dataset, indexes)
