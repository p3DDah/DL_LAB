import datetime
import os
import shutil


def get_savedir(model, dataset, max_files):
    num_files = max_files
    """Get unique saving directory name."""
    dt = datetime.datetime.now()
    date = dt.strftime("%d_%m_%H_%M")
    save_dir = os.path.join(
        "savings", model, dataset, date
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    files = sorted(os.listdir(os.path.join("savings", model, dataset)), reverse=True)
    if len(files) > num_files:
        for i,elem in enumerate(files):
            shutil.rmtree(os.path.join(os.path.join("savings", model, dataset),elem)) if i >= num_files else None
    return save_dir

def count_params(model):
    """Count total number of trainable parameters in model"""
    total = 0
    for x in model.parameters():
        if x.requires_grad:
            res = 1
            for y in x.shape:
                res *= y
            total += res
    return total
