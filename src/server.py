from copy import deepcopy
import random

import torch
import torch.nn as nn
import torch.optim as optim

from .client import Client
from .models import *
from .utils import load_cifar, run_accuracy


class Server:
    def __init__(self, device, data_config, model_config, optim_config, fed_config):
        self.device = device
        self.clients = []

        # DATASET CONFIGURATION
        self.trainset = load_cifar(name=data_config["dataset_name"], train=True)
        self.testset = load_cifar(name=data_config["dataset_name"], train=False)
        self.trainset_size = len(self.trainset)
        self.testset_size = len(self.testset)
        self.num_classes = len(self.trainset.classes)
        self.IID = data_config["IID"]
        self.global_batch_size = data_config["global_batch_size"]

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

    def init_clients(self):
        for i in range(self.num_clients):
            if self.IID:
                indexes = list(range(i, self.trainset_size, self.num_clients))
                trainset_i = torch.utils.data.Subset(self.trainset, indexes)
                client = Client(i, self.device, self.local_epochs, self.client_batch_size, trainset_i,
                                model_config=self.model_config, optim_config=self.optim_config)
                self.clients.append(client)
            else:
                raise NameError("NON-IID not yet implemented")

    def run_training(self):
        if len(self.clients) == 0:
            raise NameError("Clients NOT initialized")

        self.global_net.to(self.device)

        for t in range(self.num_rounds):
            print(f"ROUND {t+1}")

            # Save state at round t
            state_t = deepcopy(self.global_net.state_dict())

            # Get the selected clients for this round
            num_selected_clients = int(max(min(self.num_clients,
                                               random.gauss(self.avg_clients_rounds * self.num_clients,
                                                            self.std_clients_rounds * self.num_clients)), 1))
            selected_clients = random.sample(self.clients, num_selected_clients)
            num_samples = sum(c.trainset_size for c in selected_clients)

            if self.std_clients_rounds != 0:
                print(f"{num_selected_clients} clients selected")

            # Run update on each client
            for client in selected_clients:
                client.client_update(state_t)

            print("Server (BEFORE AVG) ->", end=' ')
            self.run_weighted_clients_accuracy()

            # AVERAGING
            # reset to 0 all global_net parameters
            for layer in self.global_net.parameters():
                nn.init.zeros_(layer)

            # do the average
            for client in selected_clients:
                for key in self.global_net.state_dict().keys():
                    tensor = client.net.state_dict()[key]
                    weight = client.trainset_size / num_samples
                    self.global_net.state_dict()[key] += weight * tensor

            print("Server (AFTER AVG)  ->", end=' ')
            self.run_weighted_clients_accuracy(state_dict=self.global_net.state_dict())

    def run_weighted_clients_accuracy(self, state_dict=None):
        accuracy = 0
        loss = 0
        for client in self.clients:
            client_accuracy, client_loss = client.train_accuracy(state_dict=state_dict)
            weight = client.trainset_size / self.trainset_size
            accuracy += weight * client_accuracy
            loss += weight * client_loss

        print(f'Train (weighted clients): Loss {loss:.3f} | Accuracy = {accuracy:.3f}')

    def run_testing(self):
        criterion = eval(self.model_config["criterion"])()
        accuracy, loss = run_accuracy(device=self.device, dataset=self.testset,
                                      batch_size=self.global_batch_size, net=self.global_net,
                                      criterion=criterion)
        print(f'Test Set: Loss {loss:.3f} | Accuracy = {accuracy:.3f}')
