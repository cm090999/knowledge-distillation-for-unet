from unet import unet_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from samDataset import SAM_Dataset
from diode_dataset_loader import diode_dataset
import loss
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm

# from backboned_unet import Unet
import segmentation_models_pytorch as smp
# loss functions
import pytorch_3dunet.pytorch3dunet.unet3d.losses as lsses


class UnorderedMultiLabelImageSegmentationLoss(nn.Module):
    def __init__(self, num_iters=20, reg=0.1):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.num_iters = num_iters
        self.reg = reg

    def sinkhorn(self, Q):
        Q = Q - Q.max(dim=1, keepdim=True).values
        Q = torch.exp(torch.clamp(Q / self.reg, -10, 10))
        Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)
        K, _ = Q.shape

        u = torch.zeros_like(Q)
        r = torch.ones((K,)).to(Q.device) / K
        c = torch.ones((K,)).to(Q.device) / K

        for _ in range(self.num_iters):
            u = torch.sum(Q, dim=1) + 1e-8
            Q = Q * (r / u).view((K, 1))
            Q = Q * (c / (torch.sum(Q, dim=0) + 1e-8)).view((1, K))

        return Q

    def forward(self, preds, targets):
        batch_size, num_classes, hight, width = targets.shape

        total_loss = 0

        for batch in range(batch_size):
            cost_matrix = []

            for pred_class in range(num_classes):
                pred_for_class = preds[batch, pred_class, :, :]

                class_losses = []

                for target_class in range(num_classes):
                    target_for_class = targets[batch, target_class, :, :]

                    loss = self.bce_loss(pred_for_class.float(), target_for_class.float())
                    class_losses.append(loss.sum())
                
                class_losses = torch.stack(class_losses)
                cost_matrix.append(class_losses)

            cost_matrix = torch.stack(cost_matrix)

            P = self.sinkhorn(-cost_matrix)  # compute assignment matrix using sinkhorn
            total_loss = total_loss + torch.sum(P * cost_matrix)  # compute total loss

        return total_loss / (hight * width)


# https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987/4
def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def convert_to_binary_mask(output):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # output is a tensor of shape [batch_size, num_classes, height, width]
    _, max_indices = output.max(dim=1)
    # max_indices is now of shape [batch_size, height, width]
    # and contains the index of the max class at each pixel

    # Now we'll convert max_indices to a binary mask for each class
    batch_size, height, width = max_indices.shape
    num_classes = output.shape[1]
    binary_masks = torch.zeros(batch_size, num_classes, height, width, device=output.device)
    binary_masks.scatter_(1, max_indices.unsqueeze(1), 1).to(device)
    return binary_masks

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # studentModel = Unet(backbone_name='resnet50', classes=10)

    # Define student network to train 
    channel_depth = 16
    n_channels = 3
    n_classes = 10
    # studentModel = unet_model.UNet(channel_depth=channel_depth,n_channels=n_channels,n_classes=n_classes)
    # studentModel = unet_model.UNet_ResNet34()
    studentModel = smp.Unet('resnet34', 
                            classes=10).to(device=device)
    # studentModel = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    #                         n_channels=n_channels, n_classes=n_classes, channel_depth=32, pretrained=True)

    # Define Transformations
    common_transforms = transforms.Compose([
                                            transforms.Resize(size=(320, 256),antialias=False)
                                           ])
    input_transforms = transforms.Compose([ 
                                            # transforms.ToTensor(),
                                            transforms.ConvertImageDtype(torch.float32),
                                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                            # transforms.Normalize(mean=[1], std=[1]),
                                            transforms.RandomAdjustSharpness(0.5)
                                           ])

    # Get dataset
    validationFraction = 0.1
    # datasetpath = 'dataset/samMasks'
    datasetpath = 'dataset/diodeMasks'
    dataset = SAM_Dataset(dataPath=datasetpath, common_transform=common_transforms,input_transform=input_transforms)

    # # Define the indices of the first 10 images
    # subset_indices = list(range(10))
    # # Create the subset dataset
    # subset_dataset = Subset(dataset, subset_indices)
    # dataset = subset_dataset

    dataset_split = train_val_dataset(dataset,val_split=validationFraction)
    trainingSet = dataset_split['train']
    validationSet = dataset_split['val']


    # Define Training parameters
    n_epochs = 50
    batchsize = 24
    lr = 15e-5

    # Get dataloaders
    trainingLoader = DataLoader(dataset=trainingSet,
                                batch_size=batchsize,
                                shuffle=True)
    
    validationLoader = DataLoader(dataset=validationSet,
                                  batch_size=1,
                                  shuffle=True)


    # Define Cost function and optimizer
    optimizer = torch.optim.Adam(studentModel.parameters(), lr=lr)
    # loss_function = UnorderedMultiLabelImageSegmentationLoss(num_iters=20, reg=0.1)
    loss_function = lsses.BCEDiceLoss(alpha=1,
                                      beta=1)


    # Main training loop
    best_val = 100000

    for epoch in range(n_epochs):
        print("Running Epoch " + str(epoch+1) + " of " + str(n_epochs))

        # Training Epoch
        print("Running Training")
        studentModel.train()
        avg_tr_loss = 0
        batches_tr = 0

        tqdm_console = tqdm(total=len(trainingLoader),desc='Train')

        with tqdm_console:
            tqdm_console.set_description_str('Epoch: {:03d}|{:03d}'.format(epoch+1,n_epochs))
            for i, batch in enumerate(trainingLoader):
                input = batch[0].to(device)
                gt_output = batch[1].to(device)
                output = studentModel(input)
                #binarised_output = convert_to_binary_mask(output)

                # Calculate loss
                loss_1 = loss_function(output,gt_output.type(torch.float))
                # pixelWise_loss = 
                # coherence_loss = 

                # Backpropagation 
                # clear previous gradients, compute gradients of all variables wrt loss
                optimizer.zero_grad()
                loss_1.backward()

                # performs updates using calculated gradients
                optimizer.step()

                avg_tr_loss += loss_1.item()
                batches_tr += 1

                tqdm_console.set_postfix_str("loss:{:.3f}".format(loss_1.item()))
                tqdm_console.update()

        

        print("Average Training Loss: " + str((avg_tr_loss/batches_tr)))



        # Validation Epoch
        print("Running Validation")
        studentModel.eval()
        avg_val_loss = 0
        batches = 0
        for i, batch in enumerate(validationLoader):
            input = batch[0].to(device)
            gt_output = batch[1].to(device)
            output = studentModel(input)

            # Calculate loss
            # loss_1 = loss.pixel_wise_loss(student_output=output,teacher_output=gt_output)
            loss_1 = loss_function(output,gt_output.type(torch.float))

            avg_val_loss += loss_1.item()
            batches += 1

        print("Average Validation Loss: " + str(avg_val_loss/batches))

        # Save model parameters if validation cost is best
        if avg_val_loss < best_val:
            torch.save(studentModel.state_dict(), 'checkpoint/best_model.pth') 

        # Save checkpoint if best
        