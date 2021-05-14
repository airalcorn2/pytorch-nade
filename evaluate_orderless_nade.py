import numpy as np
import shutil
import torch
import yaml

from PIL import Image
from settings import *
from torch import nn
from train_orderless_nade import init_datasets, init_model

JOB = "20210514141818"
JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"


def test_nade():
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))

    # Initialize datasets.
    (_, _, _, _, _, test_loader) = init_datasets(opts)

    # Initialize model.
    device = torch.device("cuda:0")
    model = init_model(opts).to(device)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model.eval()

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    test_loss_best_valid = 0.0
    with torch.no_grad():
        n_test = 0
        for (batch_idx, test_tensors) in enumerate(test_loader):
            print(batch_idx)
            masks = torch.zeros_like(test_tensors["mask"])
            test_tensors["mask"] = masks
            n_pix = len(masks[0])
            for n_cond in range(n_pix):
                masks[:, :n_cond] = 1
                preds = model(test_tensors).flatten()
                labels = test_tensors["pixels"].flatten().to(device)
                losses = criterion(preds[n_cond::n_pix], labels[n_cond::n_pix])
                test_loss_best_valid += losses.mean().item()
                n_test += 1

    test_loss_best_valid /= n_test
    print(test_loss_best_valid)


def generate_samples():
    opts = yaml.safe_load(open(f"{JOB_DIR}/{JOB}.yaml"))

    # Initialize model.
    device = torch.device("cuda:0")
    model = init_model(opts).to(device)
    model.load_state_dict(torch.load(f"{JOB_DIR}/best_params.pth"))
    model.eval()

    home_dir = os.path.expanduser("~")
    os.makedirs(f"{home_dir}/results", exist_ok=True)

    samples = 100
    img_size = 28
    tensors = {
        "pixels": torch.zeros(samples, img_size ** 2),
        "mask": torch.zeros(samples, img_size ** 2),
    }
    for pix_idx in range(img_size ** 2):
        with torch.no_grad():
            preds = model(tensors)

        tensors["pixels"][:, pix_idx] = torch.bernoulli(
            torch.sigmoid(preds[:, pix_idx])
        )
        tensors["mask"][:, pix_idx] = 1

    for sample in range(samples):
        img_arr = np.zeros((img_size, img_size), dtype="uint8")
        for pix_idx in range(img_size ** 2):
            (row, col) = (pix_idx // img_size, pix_idx % img_size)
            img_arr[row, col] = tensors["pixels"][sample, pix_idx].int().item()

        img = Image.fromarray(255 * img_arr)
        img.save(f"{home_dir}/results/{sample}.jpg")

    shutil.make_archive(f"{home_dir}/results", "zip", f"{home_dir}/results")
    shutil.rmtree(f"{home_dir}/results")
