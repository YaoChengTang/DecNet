import json

from loader.KITTI15Mask import KITTI15Mask
from loader.SceneflowMask import SceneflowMask
from loader.DrivingStereoMask import DrivingStereoMask
from loader.MiddleburyMask import MiddleburyMask

def get_loader(name):
    """get_loader

    :param name:
    """
    print(name.lower())
    return {
        'kitti15mask': KITTI15Mask,
        'sceneflowmask': SceneflowMask,
        'drivingstereomask': DrivingStereoMask,
        'middleburymask': MiddleburyMask,
    }[name.lower()]


def get_data_path(name, config_file='config.json'):
    """get_data_path
    
    :param name:
    :param config_file:
    """
    data = json.load(open(config_file))
    return data[name.lower()]['data_path']
