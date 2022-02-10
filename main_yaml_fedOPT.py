import torch
import yaml

from src.fedOPT.server import Server
from src.logger_manager import LoggerHandler

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CONFIG = "config_Adam"

LOG_PATH = ""
LOG_NAME = f"{CONFIG}.log"

if __name__ == "__main__":    
    print(DEVICE)

    loggerManager = LoggerHandler(LOG_PATH)
    loggerManager.create_logger(LOG_NAME, LOG_NAME)
    loggerManager.set_default_logger(LOG_NAME)

    with open(f'./configs/fedOPT/{CONFIG}.yaml') as f:
        configs = list(yaml.load_all(f, Loader=yaml.FullLoader))

    data_config = configs[0]["data_config"]
    model_config = configs[1]["model_config"]
    client_optimizer = configs[2]["client_optimizer"]
    server_optimizer = configs[3]["server_optimizer"]
    fed_config = configs[4]["fed_config"]

    server = Server(DEVICE, data_config=data_config, model_config=model_config, 
                    client_optimizer=client_optimizer, server_optimizer=server_optimizer,
                    fed_config=fed_config, logger=loggerManager)

    server.init_clients()

    # TRAINING
    server.run_training(print_acc=True)

    # TESTING
    server.run_testing()
