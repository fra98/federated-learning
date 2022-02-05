from copy import deepcopy
import random
import numpy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from .client import Client
from .models import *
from .utils import get_class_priors, load_cifar, run_accuracy, indexes_split_IID, indexes_split_NON_IID


class Server:
    def __init__(self, device, data_config, model_config, optim_config, fed_config, logger=None):
        self.device = device
        self.clients = []

        # DATASET CONFIGURATION
        self.trainset = load_cifar(name=data_config["dataset_name"], train=True)
        self.testset = load_cifar(name=data_config["dataset_name"], train=False)
        self.trainset_size = len(self.trainset)
        self.testset_size = len(self.testset)
        self.num_classes = len(self.trainset.classes)
        self.class_priors = get_class_priors(self.num_classes, self.trainset.targets)
        self.global_batch_size = data_config["global_batch_size"]
        self.std_client_samples = data_config["std_client_samples"]
        self.IID = data_config["IID"]
        self.logger = logger
        if not self.IID:
            self.alpha = data_config["alpha"]

        # MODEL CONFIGURATION
        self.model_config = model_config
        self.optim_config = optim_config
        self.global_net = eval(model_config["net"])(self.num_classes)

        # FEDERATED CONFIGURATION
        self.num_clients = fed_config["num_clients"]
        self.avg_clients_rounds = fed_config["avg_clients_rounds"]
        self.std_clients_rounds = fed_config["std_clients_rounds"]
        self.num_rounds = fed_config["num_rounds"]
        self.client_batch_size = fed_config["client_batch_size"]
        self.local_epochs = fed_config["local_epochs"]
        self.fed_IR = fed_config["fed_IR"]
        self.fed_VC = fed_config["fed_VC"]
        if self.fed_VC:
            self.virtual_client_size = self.trainset_size // self.num_clients
        else:
            self.virtual_client_size = None

    def init_clients(self):
        # Define each client training size using gaussian distribution
        avg_train_size = len(self.trainset) / self.num_clients
        clients_sizes = numpy.random.normal(avg_train_size, avg_train_size * self.std_client_samples, self.num_clients)
        delta = self.trainset_size - numpy.sum(clients_sizes) # distribute difference over all clients
        clients_sizes = (clients_sizes + delta/len(clients_sizes)).astype(int)
        self.logger.log("Client samples sizes:", clients_sizes, "total:", numpy.sum(clients_sizes), sep="\t")

        if self.IID:
            indexes = indexes_split_IID(self.num_clients, self.trainset_size)
        else:
            indexes = indexes_split_NON_IID(self.num_clients, self.num_classes, self.alpha, self.trainset, clients_sizes)

        for i in range(self.num_clients):
            trainset_i = torch.utils.data.Subset(self.trainset, indexes[i])
            client = Client(i, self.device, self.local_epochs, self.client_batch_size, trainset_i,
                            model_config=self.model_config, optim_config=self.optim_config,
                            server_class_priors=self.class_priors, virtual_client_size=self.virtual_client_size)
            self.clients.append(client)

    def run_training(self, print_acc=True):
        if len(self.clients) == 0:
            self.init_clients()

        self.global_net.to(self.device)

        for t in range(self.num_rounds):
            self.logger.log(f"ROUND {t+1}")

            # Save state at round t
            state_t = deepcopy(self.global_net.state_dict())

            # Get the selected clients for this round
            num_selected_clients = int(max(min(self.num_clients,
                                               random.gauss(self.avg_clients_rounds * self.num_clients,
                                                            self.std_clients_rounds * self.num_clients)), 1))
            if self.fed_VC:
                clients_weight = numpy.zeros((len(self.clients)))
                for i in range(len(self.clients)):
                    clients_weight[i] = self.clients[i].trainset_size
            else:
                clients_weight = numpy.ones((len(self.clients)))
            clients_weight = clients_weight / numpy.sum(clients_weight)

            selected_clients = numpy.random.choice(self.clients, num_selected_clients, replace=False, p=clients_weight)
            # selected_clients.sort(key=lambda x: x.id)
            num_samples = sum(c.trainset_size for c in selected_clients)

            if self.std_clients_rounds != 0:
                self.logger.log(f"{num_selected_clients} clients selected")

            # Run update on each client
            for client in selected_clients:
                client.client_update(state_t, fed_IR=self.fed_IR, print_acc=print_acc, fed_VC=self.fed_VC)

            if print_acc:
                self.logger.log("[BEFORE AVG]", end='\t')
                self.run_weighted_clients_accuracy()

            # AVERAGING
            # reset to 0 all global_net parameters
            for layer in self.global_net.parameters():
                nn.init.zeros_(layer)

            # do the average
            for client in selected_clients:
                if self.fed_VC:
                    # for Fed_VC we use every time the same total amount of sample per client
                    weight = 1 / len(selected_clients)
                else:
                    weight = client.trainset_size / num_samples
                
                for key in self.global_net.state_dict().keys():
                    tensor = client.net.state_dict()[key]
                    self.global_net.state_dict()[key] += weight * tensor

            self.logger.log("[AFTER AVG]", end='\t')
            self.run_testing(train=True)

    def run_weighted_clients_accuracy(self, state_dict=None):
        accuracy = 0
        loss = 0
        for client in self.clients:
            client_accuracy, client_loss = client.train_accuracy(state_dict=state_dict)
            weight = client.trainset_size / self.trainset_size
            accuracy += weight * client_accuracy
            loss += weight * client_loss

        self.logger.log(f'Weighted Clients -> Train: Loss {loss:.3f} | Accuracy = {accuracy:.3f}')

    def run_testing(self, train=False):
        if train:
            dataset = self.trainset
        else:
            dataset = self.testset
        criterion = eval(self.model_config["criterion"])()

        accuracy, loss = run_accuracy(device=self.device, dataset=dataset,
                                      batch_size=self.global_batch_size, net=self.global_net,
                                      criterion=criterion)

        if train:
            self.logger.log(f'Server -> Train: Loss {loss:.3f} | Accuracy = {accuracy:.3f}')
        else:
            self.logger.log(f'Server -> Test: Loss {loss:.3f} | Accuracy = {accuracy:.3f}')
