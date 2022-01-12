import torch
import yaml

from src.server import Server

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":    
    print(DEVICE)

    with open('./configs/config_light.yaml') as f:
        configs = list(yaml.load_all(f, Loader=yaml.FullLoader))

    data_config = configs[0]["data_config"]
    model_config = configs[1]["model_config"]
    optim_config = configs[2]["optim_config"]
    fed_config = configs[3]["fed_config"]

    server = Server(DEVICE, data_config=data_config, model_config=model_config, optim_config=optim_config,
                    fed_config=fed_config)
    server.init_clients()

    # TRAINING
    server.run_training()

    # TESTING
    server.run_testing()
