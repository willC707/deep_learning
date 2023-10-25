import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
from PIL import Image
from torch.utils.data import Dataset
import cv2


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transforms

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, f'{idx}.png')
        mask_name = os.path.join(self.mask_dir, f'{idx}.png')
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_name)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return (image, mask)

class DataLoader:
    def __init__(self):
        self.d_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )


    def get_data_loader(self, im_folder, mask_folder):
        dataset = SegmentationDataset(im_folder,mask_folder, transforms=self.d_transforms)

        data_loader = torch.utils.data.DataLoader(dataset, shuffle=False)

        return data_loader, len(dataset)