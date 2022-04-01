from functools import wraps
import os
from typing import Callable
import torch
from hprams import hprams
MODEL_KEY = 'model'
OPTIMIZER_KEY = 'optimizer'


def save_checkpoint(func) -> Callable:
    """Save a checkpoint after each iteration
    """
    @wraps(func)
    def wrapper(obj, *args, _counter=[0], **kwargs):
        _counter[0] += 1
        result = func(obj, *args, **kwargs)
        if not os.path.exists(hprams.training.checkpoints_dir):
            os.mkdir(hprams.training.checkpoints_dir)
        checkpoint_path = os.path.join(
            hprams.training.checkpoints_dir,
            'checkpoint_' + str(_counter[0]) + '.pt'
            )
        torch.save(
            {
                MODEL_KEY: obj.model.state_dict(),
                OPTIMIZER_KEY: obj.optimizer.state_dict(),
            }, 
            checkpoint_path
            )
        print(f'checkpoint saved to {checkpoint_path}')
        return result
    return wrapper
