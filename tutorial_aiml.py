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

import matplotlib.pyplot as plt;
import numpy as np;
import math
import datetime
import os
#import cv2

# =============================================================================
# TODO:
#     - data punten vervangen door plaatjes (achtergrondkleur = class)
#     - andere lagen visualiseren
#     - test class visualiseren
#     - (model opslaan/laden)
#     - video opties aanpassen
#     - draaien op cluster
# 
# =============================================================================


# Hyperparameters, general NN settings
num_epochs = 25
num_classes = 10
batch_size = 50
learning_rate = 0.001

# Data access
train_dir =  r"./data/train_diff"
test_dir =  r"./data/test_diff"
MODEL_STORE_PATH = r"./model"
start_time = datetime.datetime.now();
output_dir = f"output/{start_time.strftime('%Y-%m-%d %H.%M.%S')}"

# NN scaling params
image_width = 64    # Image width / height
no_dimens = 100    # Number of dimensions that will be reduced by UMAP to 2. Size of the 2nd fully connected layer.

# Visualization
show_after_epochs = 1
VIS_DATA = []
VIS_TARGET = []
VIS_PRED = []

TEST_DATA = []
TEST_TARGET = []
TEST_PRED = []

VIS_OUT = []
VIS_ACC = []
VIS_TEST = []


# Sketch data
train_trans = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize([image_width, image_width]),
    #transforms.RandomAffine(10, translate=(0.2,0.2), scale=(0.75,1.33), fillcolor=255),
    #transforms.RandomRotation(10),
    #transforms.RandomHorizontalFlip(),
    # translation?
    # transforms.CenterCrop(224), and rescale
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
print(classes)
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
        #print(len(VIS_DATA))
        if len(VIS_DATA)<len(trainset):
            VIS_DATA.extend(out.detach().numpy())
        else:
            TEST_DATA.extend(out.detach().numpy())
        out = self.fc2(out)
        #if len(VIS_DATA)<len(trainset):
        #    VIS_OUT.extend(out.detach().numpy())
        return out

def train_model(model):
    # Loss and optimizer
    #logsoftmax = nn.LogSoftmax(dim=1)
    #nllloss = nn.NLLLoss()
    #softmax = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        # Per batch
        total_totals = 0
        total_correct = 0
        for i, (images, labels) in enumerate(train_loader):
            # Batch i
            # Run the forward pass
            outputs = model(images)
            VIS_TARGET.extend(labels.numpy())
            #soft = softmax(outputs)
            #loss_log = logsoftmax(outputs)
            #loss = nllloss(loss_log, labels)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            
            # VIS_LOSS_TEMP.append(loss.item())
            
            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            total_totals += total
            _, predicted = torch.max(outputs.data, 1)
            VIS_PRED.extend(predicted.numpy())
            correct = (predicted == labels).sum().item()
            total_correct += correct
            accuracy = correct/total
            acc_list.append(accuracy)
            
            if (i + 1) % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (accuracy) * 100))
        #print(total_correct,'/',total_totals,'=',total_correct/total_totals)
        # VIS_LOSS.append((epoch+1,sum(VIS_LOSS_TEMP)/len(VIS_LOSS_TEMP)))
        # print(VIS_LOSS)
        if epoch == 0:
            test_model(model, epoch)
        VIS_ACC.append((epoch+1,total_correct/total_totals))
        if (epoch + 1) % show_after_epochs == 0:
            test_model(model, epoch)
            make_vis(epoch + 1)
        VIS_DATA.clear()
        VIS_TARGET.clear()
        VIS_PRED.clear()
        TEST_DATA.clear()
        TEST_TARGET.clear()
        TEST_PRED.clear()
        VIS_OUT.clear()
    return loss_list, acc_list

def test_model(model, epoch):
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            TEST_TARGET.extend(labels.numpy())
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            TEST_PRED.extend(predicted.numpy())
            total += labels.size(0)
            #print(predicted, labels)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 200 test images: {} %'.format((correct / total) * 100))
        VIS_TEST.append((epoch+1,correct/total))
    # Save the model and plot
    torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')

def plot_results(loss_list, acc_list):
    p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title=f'Performance of sketches CNN [#classes = {num_classes}, batch size = {batch_size}, #epochs = {num_epochs}, learning rate = {learning_rate}]')
    p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
    p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
    p.line(np.arange(len(loss_list)), loss_list)
    p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
    show(p)

from matplotlib.lines import Line2D
import matplotlib.colors as colors
def make_vis(epochs_passed):
    sns.set(style='white', rc={'figure.figsize':(10,8)})
    cmap=plt.cm.hsv  #Spectral #misschien .hsv (regenboog)
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name,a=0.0, b=0.9),cmap(np.linspace(0.0,0.9,10)))
    
    """
    plt.subplot(2,3,1)
    standard_embedding = umap.UMAP(random_state=42, n_neighbors=50).fit_transform(VIS_DATA)
    plt.title(f"actual train")
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=VIS_TARGET, s=1, cmap=new_cmap)
    
    plt.subplot(2,3,2)
    test_embedding = umap.UMAP(random_state=42, n_neighbors=50).fit_transform(TEST_DATA)
    plt.title(f"actual test")
    plt.scatter(test_embedding[:, 0], test_embedding[:, 1], c=TEST_TARGET, s=2, cmap=new_cmap)
    
    ax = plt.subplot(2,3,3)
    legend_colors = []
    for j in range(len(classes)):
        legend_colors.append(Line2D([0],[0],marker='o',color='w',label=classes[j],
                markerfacecolor=new_cmap(j/9),markersize=10))
    ax.legend(handles=legend_colors, loc='center')
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_locator(plt.NullLocator())
    
    #markers = ['o','v','1','p','P','X','*','d','^','_']
    #V_P_MARK = [markers[x] for x in VIS_PRED]
    
    #print(standard_embedding[:,0])
    #print('------------')
    #print(standard_embedding)
    
    plt.subplot(2,3,4)
    plt.title(f"predicted train")
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=VIS_PRED, s=1, cmap=new_cmap)
    
    plt.subplot(2,3,5)
    plt.title(f"predicted test")
    plt.scatter(test_embedding[:, 0], test_embedding[:, 1], c=TEST_PRED, s=2, cmap=new_cmap)
    
    plt.subplot(2,3,6)
    plt.title(f"accuracy")
    plt.plot(*zip(*VIS_ACC), label='train set')
    plt.plot(*zip(*VIS_TEST), label='test set')
    # plt.plot(*zip(*VIS_LOSS), label='loss train set')
    plt.legend(loc='lower right')
    
    plt.suptitle(f"Neuron activations of the sketches CNN after {epochs_passed} epochs")
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # TODO bbox_inches='tight' kan helpen
    plt.savefig(f"{output_dir}/after epoch {epochs_passed} of {num_epochs} (#c={num_classes}, bs={batch_size}, lr={learning_rate}).png")
    plt.show()
    """
    
    
    
    #markers = ['o','v','s','p','P','X','*','d','^','$c$']
    markers = ['$0$','$1$','$2$','$3$','$4$','$5$','$6$','$7$','$8$','$9$']
    fit = umap.UMAP(random_state=42, n_neighbors=25)
    standard_embedding = fit.fit_transform(VIS_DATA)
    test_embedding = fit.fit_transform(TEST_DATA)
    
    #different marker style per actual class
    #training set
    t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 = [],[],[],[],[],[],[],[],[],[]
    for i in range(len(standard_embedding)):
        if VIS_TARGET[i]==0:
            t0.append([standard_embedding[i,0], standard_embedding[i,1],VIS_PRED[i],VIS_TARGET[i]])
        if VIS_TARGET[i]==1:
            t1.append([standard_embedding[i,0], standard_embedding[i,1],VIS_PRED[i],VIS_TARGET[i]])
        if VIS_TARGET[i]==2:
            t2.append([standard_embedding[i,0], standard_embedding[i,1],VIS_PRED[i],VIS_TARGET[i]])
        if VIS_TARGET[i]==3:
            t3.append([standard_embedding[i,0], standard_embedding[i,1],VIS_PRED[i],VIS_TARGET[i]])
        if VIS_TARGET[i]==4:
            t4.append([standard_embedding[i,0], standard_embedding[i,1],VIS_PRED[i],VIS_TARGET[i]])
        if VIS_TARGET[i]==5:
            t5.append([standard_embedding[i,0], standard_embedding[i,1],VIS_PRED[i],VIS_TARGET[i]])
        if VIS_TARGET[i]==6:
            t6.append([standard_embedding[i,0], standard_embedding[i,1],VIS_PRED[i],VIS_TARGET[i]])
        if VIS_TARGET[i]==7:
            t7.append([standard_embedding[i,0], standard_embedding[i,1],VIS_PRED[i],VIS_TARGET[i]])
        if VIS_TARGET[i]==8:
            t8.append([standard_embedding[i,0], standard_embedding[i,1],VIS_PRED[i],VIS_TARGET[i]])
        if VIS_TARGET[i]==9:
            t9.append([standard_embedding[i,0], standard_embedding[i,1],VIS_PRED[i],VIS_TARGET[i]])
    
    t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 = np.array(t0), np.array(t1), np.array(t2), np.array(t3), np.array(t4), np.array(t5), np.array(t6), np.array(t7), np.array(t8), np.array(t9)
    
    plt.subplot(2,3,1)
    plt.title(f"actual train")
    plt.scatter(t0[:, 0], t0[:, 1], s=50, color=new_cmap(0.01), marker=markers[0])
    plt.scatter(t1[:, 0], t1[:, 1], s=50, color=new_cmap(0.12), marker=markers[1])
    plt.scatter(t2[:, 0], t2[:, 1], s=50, color=new_cmap(0.23), marker=markers[2])
    plt.scatter(t3[:, 0], t3[:, 1], s=50, color=new_cmap(0.34), marker=markers[3])
    plt.scatter(t4[:, 0], t4[:, 1], s=50, color=new_cmap(0.45), marker=markers[4])
    plt.scatter(t5[:, 0], t5[:, 1], s=50, color=new_cmap(0.56), marker=markers[5])
    plt.scatter(t6[:, 0], t6[:, 1], s=50, color=new_cmap(0.67), marker=markers[6])
    plt.scatter(t7[:, 0], t7[:, 1], s=50, color=new_cmap(0.78), marker=markers[7])
    plt.scatter(t8[:, 0], t8[:, 1], s=50, color=new_cmap(0.89), marker=markers[8])
    plt.scatter(t9[:, 0], t9[:, 1], s=50, color=new_cmap(1.00), marker=markers[9])
    
    plt.subplot(2,3,4)
    plt.title(f"predicted train")
    plt.scatter(t0[:, 0], t0[:, 1], s=50, color=new_cmap(0.11*t0[:,2]+0.01), marker=markers[0])
    plt.scatter(t1[:, 0], t1[:, 1], s=50, color=new_cmap(0.11*t1[:,2]+0.01), marker=markers[1])
    plt.scatter(t2[:, 0], t2[:, 1], s=50, color=new_cmap(0.11*t2[:,2]+0.01), marker=markers[2])
    plt.scatter(t3[:, 0], t3[:, 1], s=50, color=new_cmap(0.11*t3[:,2]+0.01), marker=markers[3])
    plt.scatter(t4[:, 0], t4[:, 1], s=50, color=new_cmap(0.11*t4[:,2]+0.01), marker=markers[4])
    plt.scatter(t5[:, 0], t5[:, 1], s=50, color=new_cmap(0.11*t5[:,2]+0.01), marker=markers[5])
    plt.scatter(t6[:, 0], t6[:, 1], s=50, color=new_cmap(0.11*t6[:,2]+0.01), marker=markers[6])
    plt.scatter(t7[:, 0], t7[:, 1], s=50, color=new_cmap(0.11*t7[:,2]+0.01), marker=markers[7])
    plt.scatter(t8[:, 0], t8[:, 1], s=50, color=new_cmap(0.11*t8[:,2]+0.01), marker=markers[8])
    plt.scatter(t9[:, 0], t9[:, 1], s=50, color=new_cmap(0.11*t9[:,2]+0.01), marker=markers[9])
    
    #test set
    v0, v1, v2, v3, v4, v5, v6, v7, v8, v9 = [],[],[],[],[],[],[],[],[],[]
    for i in range(len(test_embedding)):
        if TEST_TARGET[i]==0:
            v0.append([test_embedding[i,0], test_embedding[i,1],TEST_PRED[i],TEST_TARGET[i]])
        if TEST_TARGET[i]==1:
            v1.append([test_embedding[i,0], test_embedding[i,1],TEST_PRED[i],TEST_TARGET[i]])
        if TEST_TARGET[i]==2:
            v2.append([test_embedding[i,0], test_embedding[i,1],TEST_PRED[i],TEST_TARGET[i]])
        if TEST_TARGET[i]==3:
            v3.append([test_embedding[i,0], test_embedding[i,1],TEST_PRED[i],TEST_TARGET[i]])
        if TEST_TARGET[i]==4:
            v4.append([test_embedding[i,0], test_embedding[i,1],TEST_PRED[i],TEST_TARGET[i]])
        if TEST_TARGET[i]==5:
            v5.append([test_embedding[i,0], test_embedding[i,1],TEST_PRED[i],TEST_TARGET[i]])
        if TEST_TARGET[i]==6:
            v6.append([test_embedding[i,0], test_embedding[i,1],TEST_PRED[i],TEST_TARGET[i]])
        if TEST_TARGET[i]==7:
            v7.append([test_embedding[i,0], test_embedding[i,1],TEST_PRED[i],TEST_TARGET[i]])
        if TEST_TARGET[i]==8:
            v8.append([test_embedding[i,0], test_embedding[i,1],TEST_PRED[i],TEST_TARGET[i]])
        if TEST_TARGET[i]==9:
            v9.append([test_embedding[i,0], test_embedding[i,1],TEST_PRED[i],TEST_TARGET[i]])
    
    v0, v1, v2, v3, v4, v5, v6, v7, v8, v9 = np.array(v0), np.array(v1), np.array(v2), np.array(v3), np.array(v4), np.array(v5), np.array(v6), np.array(v7), np.array(v8), np.array(v9)
    
    plt.subplot(2,3,2)
    plt.title(f"actual test")
    plt.scatter(v0[:, 0], v0[:, 1], s=50, color=new_cmap(0.01), marker=markers[0])
    plt.scatter(v1[:, 0], v1[:, 1], s=50, color=new_cmap(0.12), marker=markers[1])
    plt.scatter(v2[:, 0], v2[:, 1], s=50, color=new_cmap(0.23), marker=markers[2])
    plt.scatter(v3[:, 0], v3[:, 1], s=50, color=new_cmap(0.34), marker=markers[3])
    plt.scatter(v4[:, 0], v4[:, 1], s=50, color=new_cmap(0.45), marker=markers[4])
    plt.scatter(v5[:, 0], v5[:, 1], s=50, color=new_cmap(0.56), marker=markers[5])
    plt.scatter(v6[:, 0], v6[:, 1], s=50, color=new_cmap(0.67), marker=markers[6])
    plt.scatter(v7[:, 0], v7[:, 1], s=50, color=new_cmap(0.78), marker=markers[7])
    plt.scatter(v8[:, 0], v8[:, 1], s=50, color=new_cmap(0.89), marker=markers[8])
    plt.scatter(v9[:, 0], v9[:, 1], s=50, color=new_cmap(1.00), marker=markers[9])

    plt.subplot(2,3,5)
    plt.title(f"predicted test")
    plt.scatter(v0[:, 0], v0[:, 1], s=50, color=new_cmap(0.11*v0[:,2]+0.01), marker=markers[0])
    plt.scatter(v1[:, 0], v1[:, 1], s=50, color=new_cmap(0.11*v1[:,2]+0.01), marker=markers[1])
    plt.scatter(v2[:, 0], v2[:, 1], s=50, color=new_cmap(0.11*v2[:,2]+0.01), marker=markers[2])
    plt.scatter(v3[:, 0], v3[:, 1], s=50, color=new_cmap(0.11*v3[:,2]+0.01), marker=markers[3])
    plt.scatter(v4[:, 0], v4[:, 1], s=50, color=new_cmap(0.11*v4[:,2]+0.01), marker=markers[4])
    plt.scatter(v5[:, 0], v5[:, 1], s=50, color=new_cmap(0.11*v5[:,2]+0.01), marker=markers[5])
    plt.scatter(v6[:, 0], v6[:, 1], s=50, color=new_cmap(0.11*v6[:,2]+0.01), marker=markers[6])
    plt.scatter(v7[:, 0], v7[:, 1], s=50, color=new_cmap(0.11*v7[:,2]+0.01), marker=markers[7])
    plt.scatter(v8[:, 0], v8[:, 1], s=50, color=new_cmap(0.11*v8[:,2]+0.01), marker=markers[8])
    plt.scatter(v9[:, 0], v9[:, 1], s=50, color=new_cmap(0.11*v9[:,2]+0.01), marker=markers[9])


    #legenda
    ax = plt.subplot(2,3,3)
    legend_colors = []
    for j in range(len(classes)):
        legend_colors.append(Line2D([0],[0],marker=markers[j],color='w',label=classes[j],
                markerfacecolor=new_cmap(j/9),markersize=10))
    ax.legend(handles=legend_colors, loc='center')
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_locator(plt.NullLocator())

    plt.subplot(2,3,6)
    plt.title(f"accuracy")
    plt.plot(*zip(*VIS_ACC), label='train set')
    plt.plot(*zip(*VIS_TEST), label='test set')
    # plt.plot(*zip(*VIS_LOSS), label='loss train set')
    plt.legend(loc='lower right')
    
    plt.suptitle(f"Neuron activations of the sketches CNN after {epochs_passed} epochs")
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    plt.savefig(f"{output_dir}/after epoch {epochs_passed} of {num_epochs} (#c={num_classes}, bs={batch_size}, lr={learning_rate}).png")
    
    plt.show()

# =============================================================================
# def make_video():
#     # Uncomment for manual test output_dir = "output/testimages"; # "output/2019-05-28 11.11.28";
#     images = []
#     for filename in os.listdir(output_dir):
#         img = cv2.imread(output_dir + "/" + filename)
#         height, width, layers = img.shape
#         size = (width,height)
#         images.append(img)
#     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#     video_dir = f"{output_dir}/total {num_epochs} passed (#c={num_classes}, bs={batch_size}, lr={learning_rate}).avi"
#     fps = 20
#     out = cv2.VideoWriter(video_dir, fourcc, fps, size)
#     for i in range(len(images)):
#         print(f"written image {i}")
#         out.write(images[i])
#     out.release()
# =============================================================================

if __name__ == '__main__':
    show_images();
    m = ConvNet()
    losses, accuracies = train_model(m)
    # Load a model instead: m = torch.load(MODEL_STORE_PATH + 'conv_net_model.ckpt')
    test_model(m, num_epochs)
    plot_results(losses, accuracies)
    #make_video()
