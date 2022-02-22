import os
from .dataset import Dataset
from .gtzan import GTZAN
from .magnatagatune import MAGNATAGATUNE


def get_dataset(dataset, dataset_dir, label_factor,subset, download=False):

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if dataset == "gtzan":
        d = GTZAN(root=dataset_dir, download=download, subset=subset)
    elif dataset == "magnatagatune":
        d = MAGNATAGATUNE(root=dataset_dir, download=download, label_factor=label_factor , subset=subset)
    else:
        raise NotImplementedError("Dataset not implemented")
    return d
