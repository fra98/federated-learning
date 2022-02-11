from copy import deepcopy
import random
import numpy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from .client import Client
from ..models import *
from ..utils import get_class_priors, load_cifar, run_accuracy, generate_clients_sizes, get_optimizer
from ..splits import indexes_split_IID, indexes_split_NON_IID

STEP_DOWN = False
STEP_SIZE = [12, 24, 50, 75, 100]   # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 0.3

class Server:
    def __init__(self, device, data_config, model_config, server_optimizer, client_optimizer, fed_config, logger=None):
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
        self.server_optimizer = server_optimizer
        self.client_optimizer = client_optimizer

        if self.fed_VC:
            self.virtual_client_size = self.trainset_size // self.num_clients
        else:
            self.virtual_client_size = None

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
                            model_config=self.model_config, optim_config=self.client_optimizer,
                            server_class_priors=self.class_priors, virtual_client_size=self.virtual_client_size,
                            logger=self.logger)
            self.clients.append(client)


    def run_training(self, state_dict=None, state_dict_opt=None, round_num=0, print_acc=True):
        if len(self.clients) == 0:
            self.init_clients()

        self.global_net.to(self.device)
        if state_dict is not None:
            self.global_net.load_state_dict(state_dict)
        self.global_net.train()     # when gloabal net does it train?
        state_t = deepcopy(self.global_net.state_dict())
        trainable_params = [p for p in self.global_net.parameters()] #if p.requires_grad]
        self.optimizer = get_optimizer(self.server_optimizer, trainable_params)

        if state_dict_opt is not None:
            self.optimizer.load_state_dict(state_dict_opt)
        scheduler = None

        if STEP_DOWN:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=STEP_SIZE, gamma=GAMMA)

        for _ in range(self.num_rounds):
            round_num += 1
            self.logger.log(f"ROUND {round_num}")
            self.optimizer.zero_grad()

            # Calculate clients weights based on how many samples they have
            if self.fed_VC:
                clients_weight = numpy.zeros((len(self.clients)))
                for i in range(len(self.clients)):
                    clients_weight[i] = self.clients[i].trainset_size
            else:
                clients_weight = numpy.ones((len(self.clients)))
            clients_weight = clients_weight / numpy.sum(clients_weight)

            # Get the selected clients for this round
            num_selected_clients = int(max(min(self.num_clients,
                                               random.gauss(self.avg_clients_rounds * self.num_clients,
                                                            self.std_clients_rounds * self.num_clients)), 1))
            selected_clients = numpy.random.choice(self.clients, num_selected_clients, replace=False, p=clients_weight).tolist()
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
                #self.run_weighted_clients_accuracy()

            # AVERAGING
            # reset to 0 all global_net parameters
            '''
            for layer in self.global_net.parameters():
                nn.init.zeros_(layer)
            '''

            # do the average
            for client in selected_clients:
                if self.fed_VC:
                    # for Fed_VC we use every time the same total amount of sample per client
                    weight = 1 / len(selected_clients)
                else:
                    weight = client.trainset_size / num_samples

                for p in range(len(trainable_params)):
                    tensor = (weight * client.net_updates[p]).type(client.net_updates[p].type())
                    if trainable_params[p].grad is None:
                        trainable_params[p].grad = tensor
                    else:
                        trainable_params[p].grad += tensor

            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()

            '''
            for key in self.global_net.state_dict().keys():
                # now we have a full update delta_i
                if self.delta_init:
                    self.delta_t[key] = (1 - self.beta1) * (delta_i[key])
                else:
                    self.delta_t[key] = (self.beta1 * self.delta_t[key]) + (1 - self.beta1) * (delta_i[key])
                self.vt[key] = self.vt[key] + (self.delta_t[key] ** 2)
                state_t[key] = (state_t[key] - (self.n * self.delta_t[key] / (torch.sqrt(self.vt[key]) + self.tau))).type(tensor.type())
            self.delta_init = False
            '''
            # Calculate weighted accuracy of all clients (after clients updating, AFTER averaging)
            state_t = deepcopy(self.global_net.state_dict())
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