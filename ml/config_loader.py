import os 
from types import SimpleNamespace

import yaml

class ConfigLoader():
    def __init__(self, config_name):
        with open(os.path.join('config','default.yml'), 'r') as file:
            self.config = yaml.safe_load(file)
        with open(os.path.join('config', config_name), 'r') as file:
            self.config = {**self.config, **yaml.safe_load(file)}
        
    def get_config(self):
        return SimpleNamespace(**self.config)