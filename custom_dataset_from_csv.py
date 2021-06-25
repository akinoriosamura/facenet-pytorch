import pandas as pd
import numpy as np
from PIL import Image
import os

from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets


class CustomDatasetFromCsvLocation(Dataset):
    def __init__(self, csv_path, data_dir, phase, transformations):
        """
        Custom dataset example for reading image locations and labels from csv
        but reading images from files
        Args:
            csv_path (string): path to csv file
        """
        # Transforms
        # Read the csv file
        # test, validation, training
        self.phase = phase
        self._data_info = pd.read_csv(csv_path, header=None)
        self.data_info = self._data_info[self._data_info[0] == self.phase]
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        self.image_arr = np.array([os.path.join(data_dir, os.path.basename(image_path)[:-4] + '.jpg') for image_path in self.image_arr])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 2])
        self.label_arr = np.where(self.label_arr == '男顔', 1, 0)
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.transformations = transformations

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        img_as_tensor = self.transformations(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


if __name__ == "__main__":
    # Call dataset
    csv_path = './labels_2021-06-07-classification_imahira.csv'
    data_dir = '../../dataset/puri_dataset/puri_face_Ariel_TOY_1024/'
    transformations = transforms.Compose([transforms.ToTensor(), transforms.Resize((512, 512))])
    custom_mnist_from_images =  \
        CustomDatasetFromCsvLocation(
            csv_path,
            data_dir,
            'validation',
            transformations
        )
    print(len(custom_mnist_from_images))