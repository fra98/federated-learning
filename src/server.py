from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from .client import Client
from .models import *

class Server():
  def __init__(self, device, data_config={}, model_config={}, optim_config={}, fed_config={}):
    self.device = device 
    self.clients = []

    # DATASET CONFIGURATION
    self.num_classes = data_config["num_classes"]
    self.trainset = data_config["trainset"]
    self.testset = data_config["testset"]
    self.trainset_size = len(self.trainset)
    self.testset_size = len(self.testset)
    self.IID = data_config["IID"]
    self.global_batch_size = data_config["global_batch_size"]

    # MODEL CONFIGURATION
    self.model_config = model_config
    self.optim_config = optim_config
    self.global_net = eval(model_config["net"])()

    # FEDERATED CONFIGURATION
    self.num_clients = fed_config["num_clients"]
    self.num_rounds = fed_config["num_rounds"]
    self.clients_fraction = fed_config["clients_fraction"]
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

      # Run update on each client 
      for client in self.clients:
        client.client_update(state_t)

      # AVERAGING

      # Reset to 0 all gloabl_net parameters
      for layer in self.global_net.parameters():
        nn.init.zeros_(layer)

      # Do the average
      for client in self.clients:
        for key in self.global_net.state_dict().keys():
          tensor = client.net.state_dict()[key]
          weighted_tensor = tensor * (client.trainset_size / self.trainset_size)
          self.global_net.state_dict()[key] += weighted_tensor

  def run_testing(self):
    testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.global_batch_size, shuffle=False, num_workers=2)
    self.global_net.train(False)
    criterion = eval(self.model_config["criterion"])()
    val_loss_epoch = 0
    numCorr = 0
    val_samples = self.testset_size
    val_steps = len(testloader)

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for j, data in enumerate(testloader):
            # calculate outputs by running images through the network
            images, labels = data[0].to(self.device), data[1].to(self.device)
            outputs = self.global_net(images)

            # the class with the highest energy is what we choose as prediction
            val_loss = criterion(outputs, labels)
            val_loss_step = val_loss.data.item()
            val_loss_epoch += val_loss_step
            _, predicted = torch.max(outputs.data, 1)
            numCorr += torch.sum(predicted == labels.data).data.item()
        
        val_accuracy = (numCorr / val_samples) * 100
        avg_val_loss = val_loss_epoch / val_steps

    print(f'Test Set: Loss {avg_val_loss:.3f} | Accuracy = {val_accuracy:.3f}')