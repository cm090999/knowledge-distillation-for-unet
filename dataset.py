import torch
from torch.utils.data import Dataset
from data_augmentation import load_data, split_squares
import random
import numpy as np
import os
from torchvision.io import read_image
import matplotlib.pyplot as plt
import cv2

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

class listDataset(Dataset):

    def __init__(self, ids, shuffle = True, transform = None, num_workers = 4):
        #data loading
        random.shuffle(ids)

        self.nSamples = len(ids)
        self.lines = ids
        self.transform = transform
        #self.batch_size = batch_size -> Bs is mentioned in dataloader
        self.num_workers = num_workers
         

    def __getitem__(self, index):
        assert index <= len(self), 'Error: index out of bound'
        
        img_path = self.lines[index]
        img, gt = load_data(img_path)

        if self.transform is not None:
            img = self.transform(img)

        #img = np.array(img)
        gt = np.array(gt)
        #split to squares
        gt = np.expand_dims(gt, axis = 2)
        gt = gt.transpose(2, 0, 1)

        #returning imgs and gts as a tensor
        #(1, 3, 320, 320), (1, 1, 320, 320)  - for batch_size = 1
        gt = np.array(gt)
        i = img.float()
        g = torch.from_numpy(gt).float()
        return i, g
            

            

    def __len__(self):
        #len(dataset)
        return self.nSamples
    
class ReDWeb_V1_Dataset(Dataset):
    def __init__(self, datasetPath: str, val_fraction = 0.1) -> None:
        super().__init__()

        self.datasetPath = datasetPath
        self.depthPath = datasetPath + '/RDs/'

        self.images = sorted([os.path.join(self.depthPath, img) for img in os.listdir(self.depthPath)])

    def __getitem__(self, index):
        imagepth = self.images[index]
        image = read_image(imagepth)
        return image
    



def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)



         
if __name__ == "__main__":
    datasetPath = 'dataset/ReDWeb_V1'
    checkpoint_path = "checkpoint/sam_vit_b_01ec64.pth"
    depth_dataset = ReDWeb_V1_Dataset(datasetPath=datasetPath)

    testimg = depth_dataset[0].detach().numpy()[0,:,:]
    testimg_3 = cv2.merge((testimg,testimg,testimg))

    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    mask_generator = SamAutomaticMaskGenerator(model=sam,
                                                points_per_side=32,
                                                pred_iou_thresh=0.96,
                                                stability_score_thresh=0.92,
                                                crop_n_layers=1,
                                                crop_n_points_downscale_factor=2,
                                                min_mask_region_area=100,  # Requires open-cv to run post-processing)
                                                )
    masks = mask_generator.generate(testimg_3)

    plt.figure(figsize=(20,20))
    plt.imshow(testimg_3)
    show_anns(masks)
    plt.axis('off')
    plt.show()

    print("end")