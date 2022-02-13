import torch
import yaml

from src.fedAVG.server import Server
from src.logger_manager import LoggerHandler

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CONFIG = "config_light"

LOG_PATH = ""
LOG_NAME = f"{CONFIG}.log"

if __name__ == "__main__":    
    print(DEVICE)

    loggerManager = LoggerHandler(LOG_PATH)
    loggerManager.create_logger(LOG_NAME, LOG_NAME)
    loggerManager.set_default_logger(LOG_NAME)

    with open(f'./configs/fedAVG/{CONFIG}.yaml') as f:
        configs = list(yaml.load_all(f, Loader=yaml.FullLoader))

    data_config = configs[0]["data_config"]
    model_config = configs[1]["model_config"]
    optim_config = configs[2]["optim_config"]
    fed_config = configs[3]["fed_config"]

    server = Server(DEVICE, data_config=data_config, model_config=model_config, optim_config=optim_config,
                    fed_config=fed_config, logger=loggerManager)
    
    server.init_clients()

    # TRAINING
    server.run_training(print_acc=True)

    # TESTING
    server.run_testing()
