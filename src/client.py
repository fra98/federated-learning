import torch
import torch.nn as nn
import torch.optim as optim

from .models import LeNet5_MOD

class Client():
  def __init__(self, id, device, local_epochs, batch_size, trainset):
    self.id = id
    self.device = device
    self.batch_size = batch_size
    self.local_epochs = local_epochs
    self.trainset = trainset
    self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    self.trainset_size = len(trainset)
    self.net = None

  def init_net(self, state_dict):
    self.net = LeNet5_MOD()
    self.net.to(self.device)
    self.net.load_state_dict(state_dict)

  def client_update(self, criterion, optimizer):
    if self.net is None:
      raise NameError("Client Net NOT initialized")

    trainSamples = self.trainset_size - (self.trainset_size % self.batch_size)
    iterPerEpoch = len(self.trainloader)

    for epoch in range(self.local_epochs):  # loop over the dataset multiple times
        epoch_loss = 0
        numCorrTrain = 0

        for _, data in enumerate(self.trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(self.device), data[1].to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #compute the training accuracy 
            _, predicted = torch.max(outputs.data, 1)
            numCorrTrain += torch.sum(predicted == labels.data).data.item()
            step_loss = loss.data.item()
            epoch_loss += step_loss

        avg_loss = epoch_loss / iterPerEpoch
        trainAccuracy = (numCorrTrain / trainSamples) * 100
        print(f'Client {self.id} -> Train: Epoch = {epoch+1} | Loss = {avg_loss:.3f} | Accuracy = {trainAccuracy:.3f}')