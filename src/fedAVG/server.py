from copy import deepcopy
import random
import numpy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from .client import Client
from ..models import *
from ..utils import get_class_priors, load_cifar, run_accuracy, generate_clients_sizes
from ..splits import indexes_split_IID, indexes_split_NON_IID

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
            self.clients_weights = None

    def init_clients(self):
        # Define each client training size using gaussian distribution
        clients_sizes = generate_clients_sizes(self.trainset_size, self.num_clients, self.std_client_samples)
        self.logger.log(f"Client samples sizes: {clients_sizes}, total: {numpy.sum(clients_sizes)}")

        if self.IID:
            indexes = indexes_split_IID(self.num_clients, self.num_classes, self.trainset, clients_sizes)
        else:
            indexes = indexes_split_NON_IID(self.num_clients, self.num_classes, self.alpha, self.trainset, clients_sizes)

        for i in range(self.num_clients):
            trainset_i = torch.utils.data.Subset(self.trainset, indexes[i])
            client = Client(i, self.device, self.local_epochs, self.client_batch_size, trainset_i,
                            model_config=self.model_config, optim_config=self.optim_config,
                            server_class_priors=self.class_priors, virtual_client_size=self.virtual_client_size,
                            logger=self.logger)
            self.clients.append(client)

        # Only for FedVC -> calculate clients weights based on how many samples they have                                                            
        if self.fed_VC:
            self.clients_weights = numpy.zeros((len(self.clients)))
            for i in range(len(self.clients)):
                self.clients_weights[i] = self.clients[i].trainset_size
            self.clients_weights = self.clients_weights / numpy.sum(self.clients_weights)

    def run_training(self, state_dict=None, round_num=0, print_acc=True):
        if len(self.clients) == 0:
            self.init_clients()

        self.global_net.to(self.device)
        if state_dict is not None:
            self.global_net.load_state_dict(state_dict)
        self.global_net.train()     # when gloabal net does it train?

        for _ in range(self.num_rounds):
            round_num += 1
            self.logger.log(f"ROUND {round_num}")

            # Save state at current round
            state_t = deepcopy(self.global_net.state_dict())

            # Get the selected clients for this round
            num_selected_clients = int(max(min(self.num_clients,
                                               random.gauss(self.avg_clients_rounds * self.num_clients,
                                                            self.std_clients_rounds * self.num_clients)), 1))
            selected_clients = numpy.random.choice(self.clients, num_selected_clients, replace=False, p=self.clients_weights).tolist()
            selected_clients.sort(key=lambda x: x.id)
            num_samples = sum(c.trainset_size for c in selected_clients)  # effective number of samples at current round

            if self.std_clients_rounds != 0:
                self.logger.log(f"{num_selected_clients} clients selected")

            # Run update on each client
            for client in selected_clients:
                client.client_update(state_t, fed_IR=self.fed_IR, fed_VC=self.fed_VC, print_acc=print_acc)

            # Calculate weighted accuracy of all clients (after clients updating, BEFORE averaging)
            if print_acc:
                self.logger.log("[BEFORE AVG]")
                self.run_weighted_clients_accuracy()

            # AVERAGING
            old_state = deepcopy(self.global_net.state_dict())
            for client in selected_clients:
                if self.fed_VC:
                    # for Fed_VC we use every time the same total amount of sample per client
                    weight = 1 / len(selected_clients)
                else:
                    weight = client.trainset_size / num_samples

                for key in self.global_net.state_dict().keys():
                    old_tensor = old_state[key]
                    new_tensor = client.net.state_dict()[key]
                    delta = new_tensor - old_tensor
                    self.global_net.state_dict()[key] += (weight * delta).type(old_tensor.type())

            # Calculate weighted accuracy of all clients (after clients updating, AFTER averaging)
            if print_acc:
                self.logger.log("[AFTER AVG]")
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
        
        return accuracy, loss
