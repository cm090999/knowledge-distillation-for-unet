from unet import unet_model
import torch
from samDataset import SAM_Dataset
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms


# https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987/4
def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

if __name__ == "__main__":

    # Define student network to train 
    channel_depth = 16
    n_channels = 1
    n_classes = 10
    studentModel = unet_model.UNet(channel_depth=channel_depth,n_channels=n_channels,n_classes=n_classes)

    # Define Transformations
    train_transforms = transforms.Compose([ 
                                            # transforms.ToTensor(),
                                            transforms.ConvertImageDtype(torch.float32),
                                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            transforms.Normalize(mean=[1], std=[1]),
                                            transforms.Resize(size=(320, 240),antialias=False)
                                           ]) 

    # Get dataset
    validationFraction = 0.1
    datasetpath = 'dataset/samMasks'
    dataset = SAM_Dataset(dataPath=datasetpath, transform=train_transforms)
    dataset_split = train_val_dataset(dataset,val_split=validationFraction)
    trainingSet = dataset_split['train']
    validationSet = dataset_split['val']


    # Define Training parameters
    n_epochs = 10
    batchsize = 8

    # Get dataloaders
    trainingLoader = DataLoader(dataset=trainingSet,
                                batch_size=batchsize,
                                shuffle=True)
    
    validationLoader = DataLoader(dataset=validationSet,
                                  batch_size=1,
                                  shuffle=True)


    # Define Cost function and optimizer 



    # Main training loop
    for epoch in range(n_epochs):

        # Training Epoch
        studentModel.train()
        for i, batch in enumerate(trainingLoader):
            input = batch[0] 
            gt_output = batch[1]
            output = studentModel(input)

            # Calculate loss

            # Backpropagation 



        # Validation Epoch
        studentModel.eval()
        for i, batch in enumerate(validationLoader):
            input = batch[0]
            gt_output = batch[1]
            output = studentModel(input)

            # Calculate loss 