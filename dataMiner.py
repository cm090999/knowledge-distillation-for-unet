import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data_augmentation import load_data, split_squares
import random
import numpy as np
import os
from torchvision.io import read_image
import matplotlib.pyplot as plt
import cv2

from sam import SamModel
from diode_dataset_loader import diode_dataset

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
    def __init__(self, datasetPath: str, val_fraction = 0.1, transform = None) -> None:
        super().__init__()

        self.datasetPath = datasetPath
        self.depthPath = datasetPath + '/RDs/'
        self.transform = transform

        self.images = sorted([os.path.join(self.depthPath, img) for img in os.listdir(self.depthPath)])

    def __getitem__(self, index):
        imagepth = self.images[index]
        image_1 = read_image(imagepth)

        image = image_1.repeat(3, 1, 1)

        if self.transform:
            image = self.transform(image)

        return image
    
    def __len__(self):
        return len(self.images)
    



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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms = transforms.Compose([ 
                                            # transforms.ToTensor(),
                                            # transforms.ConvertImageDtype(torch.float32),
                                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            transforms.Resize(size=(320, 240),antialias=False)
                                           ])
    
    # Diode dataset
    dataset_path = 'dataset/DIODE'
    meta_name = 'diode_dataset_loader/diode_meta.json'
    depth_dataset = diode_dataset.DIODE_Depth(meta_fname=meta_name,
                    data_root=dataset_path,
                    splits=['train','val'],
                    scene_types=['outdoor', 'indoors'],
                    transform=train_transforms)
    # testimg = depth_dataset[0]
    
    # ReDWeb_V1 dataset
    # datasetPath = 'dataset/ReDWeb_V1'
    # depth_dataset = ReDWeb_V1_Dataset(datasetPath=datasetPath, transform=train_transforms)


    # testimg = depth_dataset[0].detach().numpy()[0,:,:]
    # testimg_3 = cv2.merge((testimg,testimg,testimg))

    checkpoint_path = "checkpoint/sam_vit_b_01ec64.pth"
    sam = SamModel(checkpoint_path=checkpoint_path)
    
    dataloader = DataLoader(dataset=depth_dataset,
                            batch_size=2,
                            shuffle=False
                            )
    
    for i,batch in enumerate(dataloader):
        print('Running batch # ' + str(i) + " of " + str(len(dataloader)))
        masks = sam.runBatch(batch=batch,path='dataset/diodeMasks')

        # image0 = batch[0,:,:,:].numpy().T
        # plt.imshow(image0)
        # plt.show()
        # plt.imshow(masks[0])
        # plt.show()

    print("end")