# import training_visualisation as tv
from const import *
from vis import Visualization

import torch
import torch.nn as nn
from torch.nn import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import math
import os

VIS_DATA = []
VIS_TARGET = []
VIS_PRED = []

VIS_OUT = []

# Creating the model
class ConvNet(nn.Module):
    def __init__(self):
        self.VIS_ACC = []
        
        # W_out = (W_in - Kernel + 2*Padding)/(Stride) + 1
        super(ConvNet, self).__init__()
        # Input
        w0 = image_width

        # Layer 1 - Convolution
        k1, s1, ch1 = 5, 1, 32      # Kernel size k1, stride s1, #channels ch1
        w1 = w0                     # Desired to first keep image proportions
        p1 = int((1/2) * (s1*w1 - s1 - w0 + k1))       # Find right padding

        # Layer 1 - Pooling
        k2, s2, ch2 = 2, 2, ch1     # Keep channel size
        w2 = math.ceil(w1 / 2)      # Halve the image size, rounding up
        p2 = int((1/2) * (s2*w2 - s2 - w1 + k2))

        # Layer 2 - Convolution
        k3, s3, ch3 = 5, 1, 64
        w3 = w2                     # Desired to first keep current proportions
        p3 = int((1/2) * (s3*w3 - s3 - w2 + k3))

        # Layer 3 - Pooling
        k4, s4, ch4 = 2, 2, ch3     # Keep channel size
        w4 = math.ceil(w3 / 2)      # Halve the image size, rounding up
        p4 = int((1/2) * (s4*w4 - s4 - w3 + k4))

        # Fully connected layers 1 & 2
        fc1_size = w4 * w4 * ch4
        fc2_size = no_dimens

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, ch1, kernel_size=k1, stride=s1, padding=p1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=k2, stride=s2, padding=p2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(ch1, ch3, kernel_size=k3, stride=s3, padding=p3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=k4, stride=s4, padding=p4))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(fc1_size, fc2_size)
        self.fc2 = nn.Linear(fc2_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        vis_data = out.detach().numpy()
        out = self.fc2(out)
        return out, vis_data

def train_model(model, output_dir): 
    # Load data
    train_trans = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize([image_width, image_width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.8], std=[0.2])
    ])
    trainset = datasets.ImageFolder(
        train_dir,
        transform=train_trans
    )
    classes = trainset.classes
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True);

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []

    # Visualization object
    visu = Visualization(classes) # , VIS_DATA, VIS_TARGET, VIS_PRED, test_vis_data, TEST_TARGET, TEST_PRED, VIS_ACC, TEST_ACC

    for epoch in range(num_epochs):
        # Per batch
        total_totals = 0
        total_correct = 0
        
        for i, (images, labels) in enumerate(train_loader):
            # Batch i
            # Run the forward pass
            outputs, train_vis_data = model(images)
            visu.VIS_DATA.extend(train_vis_data)
            visu.VIS_TARGET.extend(labels.numpy())
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            total_totals += total
            _, predicted = torch.max(outputs.data, 1)
            visu.VIS_PRED.extend(predicted.numpy())
            correct = (predicted == labels).sum().item()
            total_correct += correct
            accuracy = correct/total
            acc_list.append(accuracy)
            
            if (i + 1) % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (accuracy) * 100))
        
        # Add (0,0) point for the test accuracy graph
        if epoch == 0:
            _, _, _, test_acc, _ = test_model(model)
            visu.TEST_ACC.append((epoch+1, test_acc))

        visu.VIS_ACC.append((epoch+1, total_correct/total_totals))
        
        if (epoch + 1) % show_after_epochs == 0:
            visu.TEST_DATA, visu.TEST_TARGET, visu.TEST_PRED, test_acc, images_prediction = test_model(model)
            visu.TEST_ACC.append((epoch+1, test_acc))
            # print(images_prediction)
            for image, pred, label in images_prediction:
                # print(image)
                # print(pred)
                visu.add_class_colour(image, pred, label)
            # visu.make_vis(output_dir, epoch+1)
            visu.make_label_vis(output_dir, epoch+1)
        
        visu.clear_after_epoch()
    return classes, train_loader, loss_list, acc_list

def test_model(model):
    # Load test data
    test_trans = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize([image_width, image_width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.8], std=[0.2])
    ])
    testset = datasets.ImageFolder(
        test_dir,
        transform=test_trans
    )
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False);

    # Test the model
    model.eval()
    test_vis_data = []
    test_vis_actual = []
    test_vis_pred = []

    with torch.no_grad():
        correct = 0
        total = 0
        images_prediction = []

        # Per batch
        for images, labels in test_loader:
            test_vis_actual.extend(labels.numpy())
            outputs, test_vis_data_batch = model(images)
            test_vis_data.extend(test_vis_data_batch)
            _, predicted = torch.max(outputs.data, 1)
            test_vis_pred.extend(predicted.numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            images_prediction.extend(zip(images, predicted, labels))

        accuracy = correct / total
        print('Test accuracy of the model on the test images: {} %'.format(accuracy * 100))
    
    # Save the model and plot
    torch.save(model.state_dict(), model_dir + 'conv_net_model.ckpt')
    return test_vis_data, test_vis_actual, test_vis_pred, accuracy, images_prediction