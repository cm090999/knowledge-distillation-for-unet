import torch
import numpy as np
import os
import cv2
from pathlib import Path

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class SamModel():
    def __init__(self,checkpoint_path = "checkpoint/sam_vit_b_01ec64.pth") -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path).to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(model=self.sam,
                                                    points_per_side=32,
                                                    pred_iou_thresh=0.96,
                                                    stability_score_thresh=0.95,
                                                    crop_n_layers=1,
                                                    crop_n_points_downscale_factor=2,
                                                    min_mask_region_area=2500,  # Requires open-cv to run post-processing)
                                                    )
        
        return
    

    def runBatch(self,batch: torch.tensor, path=None):

       # Make directories 
        if path:
            # Define directory to save dataset
            output_dir1 = Path().absolute() / path / 'inputs'
            output_dir1.mkdir(exist_ok=True, parents=True)
            output_dir2 = Path().absolute() / path / 'masks'
            output_dir2.mkdir(exist_ok=True, parents=True)

        batch_size,c,w,h = batch.size()

        maskList =[]

        for i in range(batch_size):
            image = batch[i,:,:,:].numpy().T
            try:
                masks = self.mask_generator.generate(image)
            except:
                print("Exception")
                continue
            if len(masks) == 0:
                continue
            mask_img = self.createSingleIntMask(masks=masks)
            maskList.append(mask_img)

            if path:
                pathmask = path + '/masks'
                pathinput = path + '/inputs'
                self.save_images(image, pathinput)
                self.save_images(mask_img,pathmask)

        return maskList
    
    def createSingleIntMask(self,masks):

        h,w = np.shape(masks[0]['segmentation'])

        int_mask = np.zeros((h,w),dtype=np.uint8)

        for i in range(len(masks)):
            number = i+1
            int_mask[masks[i]['segmentation']] = number

        return int_mask
    
    def save_images(self,image, path):

        _, _, files = next(os.walk(path))
        file_count = len(files)

        fileName = str(file_count).zfill(5) + '.png'

        cv2.imwrite(filename=path+'/'+fileName, img=image)

        return
    
    
