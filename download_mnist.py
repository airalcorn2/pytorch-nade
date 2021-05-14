import numpy as np
import urllib.request


def download_mnist():
    # Adapted from: https://github.com/yoonholee/pytorch-vae/blob/master/data_loader/fixed_mnist.py.
    url_prefix = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist"
    print("Downloading binarized MNIST...")
    data = {}
    for dataset in ["train", "valid", "test"]:
        filename = f"binarized_mnist_{dataset}.amat"
        url = f"{url_prefix}/{filename}"
        print(f"Downloading from: {url}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Saved to {filename}.")
        with open(filename) as f:
            arr = []
            for line in f:
                arr.append([int(pix) for pix in line.split()])

            arr = np.array(arr, dtype="int8").reshape(-1, 28, 28)
            # To plot one use: Image.fromarray(255 * arr[idx]).show().
            data[dataset] = arr

    data["train"] = np.concatenate([data["train"], data["valid"]])
    np.save("mnist_train.npy", data["train"])
    np.save("mnist_test.npy", data["test"])
