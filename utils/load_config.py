"""
utility functions
"""
import os
import yaml
import numpy as np


EPS = 1e-8
np.random.seed(0)


def _make_paths_absolute(dir_, config):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Args:
        dir_ (str): The path of yaml configuration file.
        config (dict): The yaml for configuration file.

    Returns:
        The configuration information in dict format.
    """
    for key in config.keys():
        if key.endswith("_path"):
            config[key] = os.path.join(dir_, config[key])
            config[key] = os.path.abspath(config[key])
        if isinstance(config[key], dict):
            config[key] = _make_paths_absolute(dir_, config[key])
    return config


def load_yaml_config(file_path):
    """
    Load a YAML configuration file.

    Args:
        file_path (str): The path of yaml configuration file.

    Returns:
        The configuration information in dict format.
    """
    # Read YAML experiment definition file
    with open(file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    config = _make_paths_absolute(os.path.abspath('.'), config)
    return config
