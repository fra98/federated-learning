import torch
import torch.nn as nn
import torch.nn.functional as f


__models = ['LeNet5_MOD', 'LeNet5_MOD_LIGHT']

def get_net(name, num_classes, BN=False, GN=False):
    if name not in __models:
        raise NameError('Net not found/implemented')
    
    model_str = name
    if BN:
        model_str = model_str + '_BN'
    elif GN:
        model_str = model_str + '_GN'
    
    return eval(model_str)(num_classes)
            

class LeNet5_MOD(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5))
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5_MOD_LIGHT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        if num_classes != 10:
            raise NameError("LeNet5_MOD_LIGHT does not yet support num_classes different from 10")
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5_MOD_BN(nn.Module):
    def __init__(self, num_classes=10, bn_stats=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5))
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=bn_stats)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5))
        self.bn2 = nn.BatchNorm2d(64, track_running_stats=bn_stats)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.pool(f.relu(self.bn1(self.conv1(x))))
        x = self.pool(f.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class LeNet5_MOD_LIGHT_BN(nn.Module):
    def __init__(self, num_classes=10, bn_stats=False):
        super().__init__()
        if num_classes != 10:
            raise NameError("LeNet5_MOD_LIGHT does not yet support num_classes different from 10")
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5))
        self.bn1 = nn.BatchNorm2d(6, track_running_stats=bn_stats)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.bn2 = nn.BatchNorm2d(16, track_running_stats=bn_stats)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.bn1(self.conv1(x))))
        x = self.pool(f.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5_MOD_GN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5))
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5))
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=64)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.pool(f.relu(self.gn1(self.conv1(x))))
        x = self.pool(f.relu(self.gn2(self.conv2(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5_MOD_LIGHT_GN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        if num_classes != 10:
            raise NameError("LeNet5_MOD_LIGHT does not yet support num_classes different from 10")
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5))
        self.gn1 = nn.GroupNorm(num_groups=3, num_channels=6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.gn1(self.conv1(x))))
        x = self.pool(f.relu(self.gn2(self.conv2(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x