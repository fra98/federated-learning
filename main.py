import torch

from src.server import Server 

# PARAMETERS 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #'cpu' # 'cuda' or 'cpu'

# Data config
DATASET_NAME = "CIFAR10"   
IID = True
GLOBAL_BATCH_SIZE = 64 

# Model config
NET = "LeNet5_MOD_LIGHT"
CRITERION = "torch.nn.CrossEntropyLoss"
OPTIMIZER = "torch.optim.SGD"

# Optim config
LR = 0.01                 # The initial Learning Rate
MOMENTUM = 0.9            # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 4e-4       # Regularization, you can keep this at the default
#STEP_DOWN = False
#STEP_SIZE = [5, 10, 15]   # How many epochs before decreasing learning rate (if using a step-down policy)
#GAMMA = 0.1               # Multiplicative factor for learning rate step-down

# Fed config
NUM_CLIENTS = 2
NUM_ROUNDS = 2
CLIENTS_FRACTION = 1
CLIENT_BATCH_SIZE = 64
LOCAL_EPOCHS = 2



# SETUP DICTIONARIES 
data_config = {
    'dataset_name' : DATASET_NAME,
    'IID' : IID,
    'global_batch_size' : GLOBAL_BATCH_SIZE
}

model_config = {
    'net' : NET,
    'criterion' : CRITERION,
    'optimizer' : OPTIMIZER
}

optim_config = {
    'lr' : LR,
    'momentum' : MOMENTUM,          
    'weight_decay' : WEIGHT_DECAY
}

fed_config = {
    'num_clients' : NUM_CLIENTS,
    'num_rounds' : NUM_ROUNDS,
    'clients_fraction' : CLIENTS_FRACTION,
    'client_batch_size' : CLIENT_BATCH_SIZE,
    'local_epochs': LOCAL_EPOCHS   
}


if __name__ == "__main__":
    print(DEVICE)

    server = Server(DEVICE, data_config=data_config, model_config=model_config, optim_config=optim_config, fed_config=fed_config)
    server.init_clients()

    # TRAINING
    server.run_training()

    # TESTING
    server.run_testing()