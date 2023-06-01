"""Miscellaneous vision datasets."""

import os

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from src.dataloaders.base import default_data_path, SequenceDataset


class ImageNet(SequenceDataset):
    """
    .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/
        Sample-of-Images-from-the-ImageNet-Dataset-used-in-the-ILSVRC-Challenge.png
        :width: 400
        :alt: Imagenet
    Specs:
        - 1000 classes
        - Each image is (3 x varies x varies) (here we default to 3 x 224 x 224)
    Imagenet train, val and test dataloaders.
    The train set is the imagenet train.
    The val split is taken from train if a val_split % is provided, or will be the same as test otherwise
    The test set is the official imagenet validation set.

    """

    _name_ = "imagenet"
    d_input = 3
    d_output = 1000
    l_output = 0

    init_defaults = {
        "data_dir": None,
        "cache_dir": None,
        "image_size": 224,
        "val_split": None,  # currently not implemented
        "train_transforms": None,
        "val_transforms": None,
        "test_transforms": None,
        "mixup": None,  # augmentation
        "num_aug_repeats": 0,
        "num_gpus": 1,
        "shuffle": True,  # for train
        "loader_fft": False,
    }

    @property
    def num_classes(self) -> int:
        """
        Return:
            1000
        """
        return 1000

    def _verify_splits(self, data_dir: str, split: str) -> None:
        dirs = os.listdir(data_dir)

        if split not in dirs:
            raise FileNotFoundError(
                f"a {split} Imagenet split was not found in {data_dir},"
                f" make sure the folder contains a subfolder named {split}"
            )

    def prepare_data(self) -> None:
        """This method already assumes you have imagenet2012 downloaded. It validates the data using the meta.bin.
        .. warning:: Please download imagenet on your own first.
        """
        if not self.use_archive_dataset:
            self._verify_splits(self.data_dir, "train")
            self._verify_splits(self.data_dir, "val")
        else:
            if not self.data_dir.is_file():
                raise FileNotFoundError(f"""Archive file {str(self.data_dir)} not found.""")

    def setup(self, stage=None):
        """Creates train, val, and test dataset."""

        from typing import Any, Callable, List, Optional, Union

        import hydra  # for mixup
        from pl_bolts.transforms.dataset_normalizations import \
            imagenet_normalization
        from torch.utils.data import Dataset
        from torch.utils.data.dataloader import default_collate
        from torchvision.datasets import ImageFolder

        # for access in other methods
        self.imagenet_normalization = imagenet_normalization
        self.default_collate = default_collate
        self.hydra = hydra
        self.ImageFolder = ImageFolder

        if self.mixup is not None:
            self.mixup_fn = hydra.utils.instantiate(self.mixup)
        else:
            self.mixup_fn = None

        self.dir_path = self.data_dir or default_data_path / self._name_

        if stage == "fit" or stage is None:
            self.set_phase([self.image_size])

        if stage == "test" or stage is None:
            test_transforms = (self.val_transform() if self.test_transforms is None
                               else hydra.utils.instantiate(self.test_transforms))

            self.dataset_test = ImageFolder(os.path.join(self.dir_path, 'val'), transform=test_transforms)

            # # modded, override (for debugging)
            # self.dataset_test = self.dataset_val

    def set_phase(self, stage_params=[224], val_upsample=False, test_upsample=False):
        """
        For progresive learning.
        Will modify train transform parameters during training, just image size for now,
        and create a new train dataset, which the train_dataloader will load every
        n epochs (in config).

        Later, will be possible to change magnitude of RandAug here too, and mixup alpha

        stage_params: list, list of values to change.  single [image_size] for now
        """

        img_size = int(stage_params[0])

        if val_upsample:
            self.val_transforms["input_size"] = img_size

        train_transforms = (self.train_transform() if self.train_transforms is None
                            else self.hydra.utils.instantiate(self.train_transforms))
        val_transforms = (self.val_transform() if self.val_transforms is None
                            else self.hydra.utils.instantiate(self.val_transforms))

        if self.loader_fft:
            train_transforms = torchvision.transforms.Compose(
                train_transforms.transforms + [
                    torchvision.transforms.Lambda(lambda x: torch.fft.rfftn(x, s=tuple([2*l for l in x.shape[1:]])))
                ]
            )
            val_transforms = torchvision.transforms.Compose(
                val_transforms.transforms + [
                    torchvision.transforms.Lambda(lambda x: torch.fft.rfftn(x, s=tuple([2*l for l in x.shape[1:]])))
                ]
            )

        self.dataset_train = self.ImageFolder(self.dir_path / 'train',
                                            transform=train_transforms)

        if self.val_split > 0.:
            # this will create the val split
            self.split_train_val(self.val_split)
        # will use the test split as val by default
        else:
            self.dataset_val = self.ImageFolder(self.dir_path / 'val', transform=val_transforms)

        # # modded, override (for debugging)
        # self.dataset_train = self.dataset_val

        # not sure if normally you upsample test also
        if test_upsample:
            self.test_transforms["input_size"] = img_size
            test_transforms = (self.val_transform() if self.test_transforms is None
                                else self.hydra.utils.instantiate(self.test_transforms))
            self.dataset_test = self.ImageFolder(os.path.join(self.dir_path, 'val'), transform=test_transforms)
            ## modded, override (for debugging)
            # self.dataset_test = self.dataset_val

        # could modify mixup by reinstantiating self.mixup_fn (later maybe)

    def train_transform(self):
        """The standard imagenet transforms.
        .. code-block:: python
            transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """
        preprocessing = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(self.image_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                self.imagenet_normalization(),
            ]
        )

        return preprocessing

    def val_transform(self):
        """The standard imagenet transforms for validation.
        .. code-block:: python
            transforms.Compose([
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        """

        preprocessing = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(self.image_size + 32),
                torchvision.transforms.CenterCrop(self.image_size),
                torchvision.transforms.ToTensor(),
                self.imagenet_normalization(),
            ]
        )
        return preprocessing

    def train_dataloader(self, **kwargs):
        """ The train dataloader """
        if self.num_aug_repeats == 0 or self.num_gpus == 1:
            shuffle = self.shuffle
            sampler = None
        else:
            shuffle = False
            from timm.data.distributed_sampler import RepeatAugSampler
            sampler = RepeatAugSampler(self.dataset_train, num_repeats=self.num_aug_repeats)

        # calculate resolution
        resolution = self.image_size / self.train_transforms['input_size']  # usually 1.0

        return (self._data_loader(self.dataset_train, shuffle=shuffle, mixup=self.mixup_fn, sampler=sampler, resolution=resolution, **kwargs))

    def val_dataloader(self, **kwargs):    
        """ The val dataloader """
        kwargs['drop_last'] = False

        # update batch_size for eval if provided
        batch_size = kwargs.get("batch_size_eval", None) or kwargs.get("batch_size")
        kwargs["batch_size"] = batch_size

        # calculate resolution
        resolution = self.image_size / self.val_transforms['input_size']  # usually 1.0 or 0.583

        return (self._data_loader(self.dataset_val, resolution=resolution, **kwargs))

    def test_dataloader(self, **kwargs):    
        """ The test dataloader """
        kwargs['drop_last'] = False

        # update batch_size for test if provided
        batch_size = kwargs.get("batch_size_test", None) or kwargs.get("batch_size_eval", None) or kwargs.get("batch_size")
        kwargs["batch_size"] = batch_size

        # calculate resolution
        resolution = self.image_size / self.test_transforms.get("input_size", self.val_transforms['input_size'])

        return (self._data_loader(self.dataset_test, resolution=resolution, **kwargs))

    def _data_loader(self, dataset, resolution, shuffle=False, mixup=None, sampler=None, **kwargs):
        # collate_fn = (lambda batch: mixup(*self.default_collate(batch))) if mixup is not None else self.default_collate
        collate_fn = (lambda batch: mixup(*self.collate_with_resolution(batch, resolution))) if mixup is not None else lambda batch: self.collate_with_resolution(batch, resolution)

        # hacked - can't pass this this arg to dataloader, but used to update the batch_size val / test
        kwargs.pop('batch_size_eval', None)
        kwargs.pop('batch_size_test', None)

        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs,
        )

    def collate_with_resolution(self, batch, resolution):
        stuff = self.default_collate(batch)
        return *stuff, {"resolution": resolution}
