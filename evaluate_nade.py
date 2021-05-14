import numpy as np
import shutil
import sys
import torch
import yaml

from PIL import Image
from settings import *
from train_nade import init_model


def generate_samples():
    JOB = "20210514093536"
    JOB_DIR = f"{EXPERIMENTS_DIR}/{JOB}"

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    except IndexError:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    tensors = {"pixels": torch.zeros(samples, img_size ** 2)}
    for pix_idx in range(img_size ** 2):
        with torch.no_grad():
            preds = model(tensors)

        tensors["pixels"][:, pix_idx] = torch.bernoulli(
            torch.sigmoid(preds[:, pix_idx])
        )

    for sample in range(samples):
        img_arr = np.zeros((img_size, img_size), dtype="uint8")
        for pix_idx in range(img_size ** 2):
            (row, col) = (pix_idx // img_size, pix_idx % img_size)
            img_arr[row, col] = tensors["pixels"][sample, pix_idx].int().item()

        img = Image.fromarray(255 * img_arr)
        img.save(f"{home_dir}/results/{sample}.jpg")

    shutil.make_archive(f"{home_dir}/results", "zip", f"{home_dir}/results")
    shutil.rmtree(f"{home_dir}/results")
