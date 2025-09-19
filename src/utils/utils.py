import os
import sys
import yaml
import dill
import pickle
import numpy as np
from src.logging.logger import logging
from src.exception.exception import NetworkSecurityException

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, 'rb') as f:
            return yaml.safe_load(file_path)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def write_yaml_file(file_path: str, content: object, replace: bool=False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(content, f)
    except Exception as e:
        raise NetworkSecurityException(e, sys)