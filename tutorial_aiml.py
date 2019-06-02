import torch;
import torch.nn as nn;
from torch.nn import *;
from torch.utils.data import DataLoader;
import torchvision;
import torchvision.transforms as transforms;
from torchvision import datasets;

from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d

import seaborn as sns
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import matplotlib.pyplot as plt;
import numpy as np;
import math
import datetime
import os
import cv2

# Hyperparameters, general NN settings
num_epochs = 1
num_classes = 10
batch_size = 50
learning_rate = 0.01

# Data access
train_dir =  r"./data/train_sub"
test_dir =  r"./data/test_sub"
MODEL_STORE_PATH = r"./model"
start_time = datetime.datetime.now();
output_dir = f"output/{start_time.strftime('%Y-%m-%d %H.%M.%S')}"

# NN scaling params
image_width = 64    # Image width / height
no_dimens = 1000    # Number of dimensions that will be reduced by UMAP to 2. Size of the 2nd fully connected layer.

# Visualization
show_after_epochs = 1
VIS_DATA = []
VIS_TARGET = []
VIS_OUT = []

# Sketch data
train_trans = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize([image_width, image_width]),
    # transforms.RandomRotation(10),
    # transforms.RandomHorizontalFlip(),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.8], std=[0.2])
])

test_trans = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize([image_width, image_width]),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.8], std=[0.2])
])

trainset = datasets.ImageFolder(
    train_dir,
    transform=train_trans
)
testset = datasets.ImageFolder(
    test_dir,
    transform=test_trans
)
classes = trainset.classes

# Load data
train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True);
test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False);

# Show random images
def show_images():
    # Get some random training images
    dataiter = iter(train_loader);
    images, labels = dataiter.next();
    # Show images
    imshow(torchvision.utils.make_grid(images));
    # Print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)));

def imshow(img):
    img = img / 2 + 0.5;    # Unnormalize
    npimg = img.numpy();
    plt.imshow(np.transpose(npimg, (1,2,0)));
    plt.show();

# Creating the model
class ConvNet(nn.Module):
    def __init__(self):
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
        VIS_DATA.extend(out.detach().numpy())
        out = self.fc2(out)
        VIS_OUT.extend(out.detach().numpy())
        return out

def train_model(model):
    # Loss and optimizer
    logsoftmax = nn.LogSoftmax(dim=1)
    nllloss = nn.NLLLoss()
    #softmax = nn.Softmax(dim=1)
    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        # Per batch
        for i, (images, labels) in enumerate(train_loader):
            # Batch i
            # Run the forward pass
            outputs = model(images)
            VIS_TARGET.extend(labels.numpy())
            #soft = softmax(outputs)
            loss_log = logsoftmax(outputs)
            loss = nllloss(loss_log, labels)
            #loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            # VIS_TARGET.extend(predicted.item())
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                            (correct / total) * 100))
        print(epoch + 1)
        if (epoch + 1) % show_after_epochs == 0:
            make_vis(epoch + 1)
        VIS_DATA.clear()
        VIS_TARGET.clear()
    return loss_list, acc_list

def test_model(model):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

    # Save the model and plot
    torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')

def plot_results(loss_list, acc_list):
    p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title=f'Performance of sketches CNN [#classes = {num_classes}, batch size = {batch_size}, #epochs = {num_epochs}, learning rate = {learning_rate}]')
    p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
    p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
    p.line(np.arange(len(loss_list)), loss_list)
    p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
    show(p)

def make_vis(epochs_passed):
    sns.set(style='white', rc={'figure.figsize':(10,8)})
    # mnist = load_digits()
    # print(mnist)
    # mnist = fetch_openml('mnist_784')
    # print(mnist.data)
    # print(VIS_DATA)
    # print(VIS_TARGET)

    neighs = [3, 10, 15, 50]
    for i in range(4):
        plt.subplot(2,2,i+1)
        standard_embedding = umap.UMAP(random_state=42, n_neighbors=neighs[i]).fit_transform(VIS_DATA)
        plt.title(f"#neighbors = {neighs[i]}")
        plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=VIS_TARGET, s=1, cmap='Spectral');

    plt.suptitle(f"Neuron activations of the sketches CNN after {epochs_passed} epochs")
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # TODO bbox_inches='tight' kan helpen
    plt.savefig(f"{output_dir}/after epoch {epochs_passed} of {num_epochs} (#c={num_classes}, bs={batch_size}, lr={learning_rate}).png")
    plt.show()

def make_video():
    output_dir = "output/testimages"; # "output/2019-05-28 11.11.28";
    images = []
    for filename in os.listdir(output_dir):
        img = cv2.imread(output_dir + "/" + filename)
        height, width, layers = img.shape
        size = (width,height)
        images.append(img)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_dir = f"{output_dir}/total {num_epochs} passed (#c={num_classes}, bs={batch_size}, lr={learning_rate}).avi"
    out = cv2.VideoWriter(video_dir, fourcc, 20, size)
    for i in range(len(images)):
        print(f"written image {i}")
        out.write(images[i])
    out.release()

if __name__ == '__main__':
    # show_images();
    # m = ConvNet()
    # losses, accuracies = train_model(m)
    # m = torch.load(MODEL_STORE_PATH + 'conv_net_model.ckpt')
    # test_model(m)
    # plot_results(losses, accuracies)
    make_video()
