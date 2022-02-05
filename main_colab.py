import torch
import yaml
import os

from src.server import Server
from src.logger_manager import LoggerHandler

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CONFIG = 'config_light'

BASE_PATH = f'./run/{CONFIG}/'
# BASE_PATH = f'/content/gdrive/MyDrive/run/{CONFIG}/'
NET_PATH = BASE_PATH + 'server.pth'
COUNTER_PATH = BASE_PATH + 'counter.txt'
LOG_NAME = f'{CONFIG}.log'

if __name__ == "__main__":    
    print(DEVICE)

    with open(f'./configs/{CONFIG}.yaml') as f:
        configs = list(yaml.load_all(f, Loader=yaml.FullLoader))
        loggerManager = LoggerHandler(BASE_PATH)
        loggerManager.create_logger(LOG_NAME, LOG_NAME)
        loggerManager.set_default_logger(LOG_NAME)

    data_config = configs[0]["data_config"]
    model_config = configs[1]["model_config"]
    optim_config = configs[2]["optim_config"]
    fed_config = configs[3]["fed_config"]

    server = Server(DEVICE, data_config=data_config, model_config=model_config, optim_config=optim_config,
                    fed_config=fed_config, logger=loggerManager)
    server.init_clients()

    # Loading current model
    state_dict = None
    round_num = 0
    if os.path.isfile(NET_PATH):
        state_dict = torch.load(NET_PATH)
        with open(COUNTER_PATH, 'r') as f:
            round_num = int(f.readline())

    # TRAINING
    server.run_training(print_acc=False, state_dict=state_dict, round_num=round_num)

    # TESTING
    server.run_testing()

    # Saving current model
    round_num += server.num_rounds
    torch.save(server.global_net.state_dict(), NET_PATH)
    torch.save(server.global_net.state_dict(), f'{BASE_PATH}/server_{round_num:03}.pth')
    with open(COUNTER_PATH, 'w') as f:
        f.write(str(round_num))
