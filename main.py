import torch

from src.server import Server
from src.logger_manager import LoggerHandler

# PARAMETERS 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 'cuda' or 'cpu'

# Data config
DATASET_NAME = "CIFAR10"   
IID = False
ALPHA = 1
GLOBAL_BATCH_SIZE = 64 
STD_CLIENT_SAMPLES = 0.2 # Std-dev of number of sample per client, relative to average number of sample

# Model config
NET = "LeNet5_MOD_LIGHT"
CRITERION = "torch.nn.CrossEntropyLoss"
OPTIMIZER = "torch.optim.SGD"

# Optim config
LR = 0.01                 # The initial Learning Rate
MOMENTUM = 0.9            # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 4e-4       # Regularization, you can keep this at the default
# STEP_DOWN = False
# STEP_SIZE = [5, 10, 15]   # How many epochs before decreasing learning rate (if using a step-down policy)
# GAMMA = 0.1               # Multiplicative factor for learning rate step-down

# Fed config
NUM_CLIENTS = 2
AVG_CLIENTS_ROUND = 1  # Average number of clients per round, relative to num_clients
STD_CLIENTS_ROUND = 0  # Std-dev of number of clients per round (0 for constant distribution), relative to num_clients
NUM_ROUNDS = 2
CLIENT_BATCH_SIZE = 64
LOCAL_EPOCHS = 2
FED_IR = True
FED_VC = True


# SETUP DICTIONARIES 
data_config = {
    'dataset_name': DATASET_NAME,
    'IID': IID,
    'alpha': ALPHA,
    'global_batch_size': GLOBAL_BATCH_SIZE,
    'std_client_samples': STD_CLIENT_SAMPLES 
}

model_config = {
    'net': NET,
    'criterion': CRITERION,
    'optimizer': OPTIMIZER
}

optim_config = {
    'lr': LR,
    'momentum': MOMENTUM,
    'weight_decay': WEIGHT_DECAY
}

fed_config = {
    'num_clients': NUM_CLIENTS,
    'avg_clients_rounds': AVG_CLIENTS_ROUND,
    'std_clients_rounds': STD_CLIENTS_ROUND,
    'num_rounds': NUM_ROUNDS,
    'client_batch_size': CLIENT_BATCH_SIZE,
    'local_epochs': LOCAL_EPOCHS,
    'fed_IR': FED_IR,
    'fed_VC': FED_VC
}

LOG_PATH="./"

if __name__ == "__main__":
    print(DEVICE)

    loggerManager = LoggerHandler(LOG_PATH)
    loggerManager.create_logger("main", "main.txt")
    loggerManager.set_default_logger("main")
    server = Server(DEVICE, data_config=data_config, model_config=model_config, optim_config=optim_config,
                    fed_config=fed_config, logger=loggerManager)

    server.init_clients()

    # TRAINING
    server.run_training()

    # TESTING
    server.run_testing()
