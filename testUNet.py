import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F

from unet import unet_model
from samDataset import ROS_Data

import segmentation_models_pytorch as smp

from sam import SamModel
from samDataset import SAM_Dataset

def apply_filter(tensor):
    # Define the filter size
    filter_size = 5

    # Pad the tensor to maintain the same output size
    padding = filter_size // 2
    padded_tensor = F.pad(tensor, (padding, padding, padding, padding), mode='constant')

    # Extract patches from the padded tensor
    patches = padded_tensor.unfold(1, filter_size, 1).unfold(2, filter_size, 1)

    # Reshape patches to a 2D tensor
    flattened_patches = patches.contiguous().view(*patches.size()[:-2], -1)

    # Find the most common value in each patch
    most_common_values, _ = flattened_patches.mode(dim=-1)

    return most_common_values

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load UNet
    channel_depth = 16
    n_channels = 1
    n_classes = 10
    # studentModel = unet_model.UNet(channel_depth=channel_depth,n_channels=n_channels,n_classes=n_classes).to(device)
    studentModel = smp.Unet('vgg16', 
                            classes=10).to(device=device)

    # Apply weights
    chpt_path =  'checkpoint/best_model.pth'
    studentModel.load_state_dict(torch.load(chpt_path))

    # Define Transformations
    test_transforms = transforms.Compose([ 
                                            # transforms.ToTensor(),
                                            transforms.ConvertImageDtype(torch.float32),
                                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            # transforms.Normalize(mean=[1], std=[1]),
                                            transforms.Resize(size=(320, 256),antialias=False)
                                           ]) 

    # Get test dataset
    test_data_path = 'dataset/ROS_sim/depth_Images_normalized'

    test_data = ROS_Data(test_data_path, transform=test_transforms)

    testloader = DataLoader(test_data,
                            batch_size=1,
                            shuffle=False)
    
    checkpoint_path_sam = "checkpoint/sam_vit_b_01ec64.pth"

    sam = SamModel(checkpoint_path=checkpoint_path_sam)
    
    for i,batch in enumerate(testloader):
        print("Running Batch " +str(i))
        output = studentModel(batch.to(device))


        logits = output.permute(0, 2, 3, 1)  # Reshape to (batch_size, height, width, 10)

        # Apply argmax to get the instance label with the highest score for each pixel
        _, instance_masks = torch.max(logits, dim=3)

        filtered_mask = apply_filter(instance_masks)
        filtered_mask = apply_filter(filtered_mask)

        sam_gt = sam.runBatch((batch * 255).type(torch.uint8))[0]


        origin_img = np.asarray(batch.squeeze().detach().cpu().numpy().T)
        masked_img = np.asarray(instance_masks.squeeze().detach().cpu().numpy().T)
        filtered_mask = np.asarray(filtered_mask.squeeze().detach().cpu().numpy().T)
        sam_gt_img = np.asarray(sam_gt)

        # Plot the images side by side
        fig, axes = plt.subplots(1, 4)
        axes[0].imshow(origin_img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(masked_img)
        axes[1].set_title("Masked Image")
        axes[1].axis("off")

        axes[2].imshow(filtered_mask)
        axes[2].set_title("filtered Masked Image")
        axes[2].axis("off")

        axes[3].imshow(sam_gt_img)
        axes[3].set_title("Mask with Sam")
        axes[3].axis("off")


        # Wait for the plot window to be closed
        plt.waitforbuttonpress()