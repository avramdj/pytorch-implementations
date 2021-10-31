import torch
import os, errno
from os import path
from datetime import datetime

def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def save_model(model):
    model_dir = "models"
    if not path.isdir(model_dir):
        os.mkdir(model_dir)
    save_path = path.join(model_dir, f"model-state-{datetime.now(tz=None)}")
    sym_path = path.join(model_dir, "latest")

    torch.save(model.state_dict(), save_path)
    symlink_force(path.basename(save_path), sym_path)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")