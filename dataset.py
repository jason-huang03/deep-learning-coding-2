from PIL import Image
import os
import numpy as np
import pickle # to store things onto disk
from typing import Any, Callable, Optional, Tuple
import torch

import torchvision
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms

testing = False

class CINIC10(VisionDataset):
    """
    Implement your dataset below.
    **Note**: 
        1. Load image with `Image.open()`
        2. Loaded image must be in RGB mode(**hint**: `img.convert("RGB")`)
        3. Images should be transformed before fed into neural networks.
    """
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root)

        self.split = split  # training set or test set
        self.transform = transform

        self.data: Any = []
        self.targets = []

        self.classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

        self._load_data()


    def _load_data(self) -> None:
        """
        Load data from cinic10 dataset and their corresponding label, the label of each class should be {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}
        """
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        # YOUR CODE HERE
        
        
        
        self.root = os.path.join(self.root, self.split)

        data_file = self.split+"_data"
        targets_file = self.split+"_targets"
        # if the data is already created as a file
        if os.path.isfile(data_file) and os.path.isfile(targets_file):
            with open(data_file, "rb") as f:
                self.data = pickle.load(f)
            with open(targets_file, "rb") as f:
                self.targets = pickle.load(f)
                self.targets = torch.tensor(self.targets, dtype=torch.int)
            return
        
        
        # if the data is not created as a file
        for index, cclass in enumerate(self.classes):
            temp_path = os.path.join(self.root, cclass)
            for _, filename in enumerate(os.listdir(temp_path)):
                if testing and _ > 9:
                    break
                image = Image.open(os.path.join(temp_path, filename)).convert('RGB')
                self.data.append(image)
                self.targets.append(index)
        
        with open(data_file, "wb") as f:
            pickle.dump(self.data, f)
        with open(targets_file, "wb") as f:
            pickle.dump(self.targets, f)
                
        self.targets = torch.tensor(self.targets, dtype=torch.int)
        


        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return transformed img and target
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        # YOUR CODE HERE
        img = self.data[index]
        target = self.targets[index]
        if self.transform:
            img = self.transform(img)

        
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        return "Split: {}".format(self.split)