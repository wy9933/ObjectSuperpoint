import yaml
from easydict import EasyDict
import os

def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == 'some path':
                # load other yaml config
                with open(new_config['some path'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)
    return config

def get_config(args):
    if args.resume:
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(cfg_path)
        args.config = cfg_path
    config = cfg_from_yaml_file(args.config)
    if not args.resume:
        save_experiment_config(args)
    return config

def save_experiment_config(args):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    os.system('cp {} {}'.format(args.config, config_path))


if __name__ == "__main__":
    cfg_file = "config/furniture.yaml"
    config = cfg_from_yaml_file(cfg_file)
    print(config.model.encoder.input_dim)