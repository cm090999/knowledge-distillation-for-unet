from torch.utils.data import Dataset, DataLoader
import torch
import os
from torchvision.io import read_image


class SAM_Dataset(Dataset):
    def __init__(self,dataPath, transform = None) -> None:
        super().__init__()

        self.pathInput = dataPath + '/inputs'
        self.pathMasks = dataPath + '/masks'

        self.transform = transform

        self.inputs = sorted([os.path.join(self.pathInput, img) for img in os.listdir(self.pathInput)])
        self.outputs = sorted([os.path.join(self.pathMasks, img) for img in os.listdir(self.pathMasks)])

        return
    
    def __getitem__(self, index):

        input = read_image(self.inputs[index])
        output = read_image(self.outputs[index])[0,:,:].unsqueeze(0)
        _,w,h = output.size()

        # Convert input to single channel
        input = input[0,:,:].unsqueeze(0)


        if self.transform:
            input = self.transform(input)

        # Expand output to include a channel for each class
        # Number of channels = Number of classes + 1
        output_c = torch.zeros((11,w,h),dtype=torch.uint8)
        n = 11
        for i in range(n):
            output_c[i,:,:] = output == i 

        output_c = output_c[1:,:,:]

        output_c = torch.swapaxes(output_c,1,2)


        return input, output_c
    
    def __len__(self):
        return len(self.inputs)


if __name__ == "__main__":
    test_data = SAM_Dataset('dataset/samMasks')
    testimages = test_data[0]
    length = len(test_data)
    print('end')
