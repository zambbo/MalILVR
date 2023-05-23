import os
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *,
    data_dir,
    batch_size,
    image_size=64,
    class_cond=False,
    deterministic=False,
):
    """
    For a dataset, create a generator over (malware-latent, kwargs) pairs.

    Each malware-latent is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a path of dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which malware latents are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    all_files = os.listdir(data_dir)
    all_files = [os.path.join(data_dir, item) for item in all_files]

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # benign : 1, malware : 0
        class_names = [item.split("/")[-1].split("_")[0] for item in all_files]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths
        self.local_classes = None if classes is None else classes

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        arr = np.load(path).astype(np.float32)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict
