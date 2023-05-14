from torch.utils.data import Dataset, DataLoader
import torch
import os
from torchvision.io import read_image


class SAM_Dataset(Dataset):
    def __init__(self,dataPath, common_transform = None, input_transform = None) -> None:
        super().__init__()

        self.pathInput = dataPath + '/inputs'
        self.pathMasks = dataPath + '/masks'

        self.inputTransform = input_transform
        self.commonTransform = common_transform

        self.inputs = sorted([os.path.join(self.pathInput, img) for img in os.listdir(self.pathInput)])
        self.outputs = sorted([os.path.join(self.pathMasks, img) for img in os.listdir(self.pathMasks)])

        return
    
    def __getitem__(self, index):

        input = read_image(self.inputs[index])
        output = read_image(self.outputs[index])[0,:,:].unsqueeze(0)
        

        # Convert input to single channel
        input = input[0,:,:].unsqueeze(0)

        if self.commonTransform:
            input = self.commonTransform(input)
            output = self.commonTransform(output)


        if self.inputTransform:
            input = self.inputTransform(input)

        _,w,h = output.size()

        # Expand output to include a channel for each class
        # Number of channels = Number of classes + 1
        output_c = torch.zeros((11,w,h),dtype=torch.uint8)
        n = 11
        for i in range(n):
            output_c[i,:,:] = output == i 

        output_c = output_c[1:,:,:]

        # output_c = torch.swapaxes(output_c,1,2)


        # to get 3 channels
        input_3 = input.repeat(3, 1, 1)  # Convert to 3-channel image



        return input_3, output_c
    
    def __len__(self):
        return len(self.inputs)
    

class ROS_Data(Dataset):
    def __init__(self, dataPath, transform = None) -> None:
        super().__init__()

        self.pathInput = dataPath

        self.transform = transform

        self.inputs = sorted([os.path.join(self.pathInput, img) for img in os.listdir(self.pathInput)])

        return
    
    def __getitem__(self, index):

        input = read_image(self.inputs[index])

        # Convert input to single channel
        input = input[0,:,:].unsqueeze(0)

        if self.transform:
            input = self.transform(input)

        # to get 3 channels
        input_3 = input.repeat(3, 1, 1)  # Convert to 3-channel image

        return input_3
    
    def __len__(self):
        return len(self.inputs)


if __name__ == "__main__":
    test_data = SAM_Dataset('dataset/samMasks')
    testimages = test_data[0]
    length = len(test_data)
    print('end')
