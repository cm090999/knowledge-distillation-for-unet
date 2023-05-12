from unet import unet_model
import torch
from samDataset import SAM_Dataset
import loss
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
                                            transforms.RandomResizedCrop(size=(320, 240),antialias=False)
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
    lr = 1e-3

    # Get dataloaders
    trainingLoader = DataLoader(dataset=trainingSet,
                                batch_size=batchsize,
                                shuffle=True)
    
    validationLoader = DataLoader(dataset=validationSet,
                                  batch_size=1,
                                  shuffle=True)


    # Define Cost function and optimizer
    optimizer = torch.optim.Adam(studentModel.parameters(), lr=lr)


    # Main training loop
    best_val = 100000
    for epoch in range(n_epochs):
        print("Running Epoch " + str(epoch+1) + " of " + str(n_epochs))

        # Training Epoch
        print("Running Training")
        studentModel.train()
        avg_tr_loss = 0
        batches_tr = 0
        for i, batch in enumerate(trainingLoader):
            input = batch[0] 
            gt_output = batch[1]
            output = studentModel(input)

            # Calculate loss
            loss_1 = loss.pixel_wise_loss(student_output=output,teacher_output=gt_output)
            # pixelWise_loss = 
            # coherence_loss = 

            # Backpropagation 
            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss_1.backward()

            # performs updates using calculated gradients
            optimizer.step()

            avg_tr_loss += loss_1
            batches_tr += 1

        

        print("Average Training Loss: " + str((avg_tr_loss/batches_tr).item()))



        # Validation Epoch
        print("Running Validation")
        studentModel.eval()
        avg_val_loss = 0
        batches = 0
        for i, batch in enumerate(validationLoader):
            input = batch[0]
            gt_output = batch[1]
            output = studentModel(input)

            # Calculate loss
            loss_1 = loss.pixel_wise_loss(student_output=output,teacher_output=gt_output)

            avg_val_loss += loss_1
            batches += 1

        print("Average Validation Loss: " + str((avg_val_loss/batches).item()))

        # Save model parameters if validation cost is best
        if avg_val_loss < best_val:
            torch.save(studentModel.state_dict(), 'checkpoint/best_model.pth') 

        # Save checkpoint if best
        