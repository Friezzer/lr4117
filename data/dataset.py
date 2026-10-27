import os
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from PIL import Image, UnidentifiedImageError
from configs.config import DatasetConfig

class RobustCelebA(datasets.CelebA):
    """
    Отказоустойчивая обертка над CelebA.
    При обнаружении поврежденного файла изображения возвращает соседний валидный сэмпл.
    """
    def __getitem__(self, index: int):
        try:
            return super().__getitem__(index)
        except (UnidentifiedImageError, OSError):
            # Переход к следующему индексу по модулю длины датасета
            next_index = (index + 1) % len(self)
            return self.__getitem__(next_index)

def get_transforms(image_size: int = 64, crop_size: int = 140) -> Tuple[transforms.Compose, transforms.Compose]:
    transform_train = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform_train, transform_test

def get_celeba_dataloaders(
    config: DatasetConfig
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, List[str], List[int]]:
    os.makedirs(config.data_root, exist_ok=True)
    transform_train, transform_test = get_transforms(config.image_size, config.crop_size)

    # Используем RobustCelebA с download=False
    train_dataset = RobustCelebA(
        root=config.data_root, 
        split='train', 
        target_type='attr', 
        transform=transform_train, 
        download=False
    )
    val_dataset = RobustCelebA(
        root=config.data_root, 
        split='valid', 
        target_type='attr', 
        transform=transform_test, 
        download=False
    )
    test_dataset = RobustCelebA(
        root=config.data_root, 
        split='test', 
        target_type='attr', 
        transform=transform_test, 
        download=False
    )

    all_attr_names = list(train_dataset.attr_names)
    target_indices = [all_attr_names.index(attr) for attr in config.target_attributes]
    target_names = [all_attr_names[i] for i in target_indices]

    # Ограничение выборки для WGAN
    wgan_samples = min(len(train_dataset), config.wgan_max_samples)
    wgan_train_dataset = Subset(train_dataset, list(range(wgan_samples)))

    use_cuda = torch.cuda.is_available()

    cls_train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers, 
        pin_memory=use_cuda, 
        drop_last=True
    )
    wgan_train_loader = DataLoader(
        wgan_train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers, 
        pin_memory=use_cuda, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers, 
        pin_memory=use_cuda
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers, 
        pin_memory=use_cuda
    )

    return cls_train_loader, wgan_train_loader, val_loader, test_loader, target_names, target_indices