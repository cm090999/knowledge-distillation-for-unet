import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from unet import unet_model
from samDataset import ROS_Data

import segmentation_models_pytorch as smp

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load UNet
    channel_depth = 16
    n_channels = 1
    n_classes = 10
    # studentModel = unet_model.UNet(channel_depth=channel_depth,n_channels=n_channels,n_classes=n_classes).to(device)
    studentModel = smp.Unet('resnet18', 
                            classes=10).to(device=device)

    # Apply weights
    chpt_path =  'checkpoint/best_model.pth'
    studentModel.load_state_dict(torch.load(chpt_path))

    # Define Transformations
    test_transforms = transforms.Compose([ 
                                            # transforms.ToTensor(),
                                            transforms.ConvertImageDtype(torch.float32),
                                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            transforms.Normalize(mean=[1], std=[1]),
                                            transforms.RandomResizedCrop(size=(320, 256),antialias=False)
                                           ]) 

    # Get test dataset
    test_data_path = 'dataset/ROS_sim/depth_Images_normalized'

    test_data = ROS_Data(test_data_path, transform=test_transforms)

    testloader = DataLoader(test_data,
                            batch_size=1,
                            shuffle=False)
    
    for i,batch in enumerate(testloader):
        print("Running Batch " +str(i))
        output = studentModel(batch.to(device))

        logits = output.permute(0, 2, 3, 1)  # Reshape to (batch_size, height, width, 10)

        # Apply argmax to get the instance label with the highest score for each pixel
        _, instance_masks = torch.max(logits, dim=3)

        origin_img = np.asarray(batch.squeeze().detach().cpu().numpy().T)
        masked_img = np.asarray(instance_masks.squeeze().detach().cpu().numpy().T)

        # Plot the images side by side
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(origin_img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        axes[1].imshow(masked_img)
        axes[1].set_title("Masked Image")
        axes[1].axis("off")
        plt.show()

        # Wait for the plot window to be closed
        plt.waitforbuttonpress()
     