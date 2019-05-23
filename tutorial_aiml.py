import torch;
import torch.nn as nn;
from torch.nn import *;
from torch.utils.data import DataLoader;
import torchvision;
import torchvision.transforms as transforms;
import matplotlib.pyplot as plt;
import numpy as np;
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np
from torchvision import datasets;
import seaborn as sns
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# Hyperparameters
num_epochs = 10;
num_classes = 5;
batch_size = 50;
learning_rate = 0.001;
train_dir =  r"./data/train_sub"
test_dir =  r"./data/test_sub"
MODEL_STORE_PATH = r"./model"
VIS_DATA = []
VIS_TARGET = []

# Sketch data
trans = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize([28, 28]),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.8], std=[0.2])
        ])

trainset = datasets.ImageFolder(
        train_dir,
        transform=trans
    )
testset = datasets.ImageFolder(
        test_dir,
        transform=trans
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
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # print("Layer2")
        # print(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        VIS_DATA.extend(out.detach().numpy())
        out = self.fc2(out)
        return out

def train_model(model):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
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
            loss = criterion(outputs, labels)
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

            #print(i, labels)

            if (i + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                            (correct / total) * 100))
        make_vis()
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
    p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='PyTorch ConvNet results')
    p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
    p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
    p.line(np.arange(len(loss_list)), loss_list)
    p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
    show(p)

def make_vis():
    sns.set(style='white', rc={'figure.figsize':(10,8)})
    # mnist = load_digits()
    # print(mnist)
    # mnist = fetch_openml('mnist_784')
    # print(mnist.data)
    # print(VIS_DATA)
    # print(VIS_TARGET)
    standard_embedding = umap.UMAP(random_state=42).fit_transform(VIS_DATA)
    plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=VIS_TARGET, s=1, cmap='Spectral');
    plt.show()
    VIS_DATA.clear()
    VIS_TARGET.clear()

if __name__ == '__main__':
    show_images();
    m = ConvNet()
    losses, accuracies = train_model(m)
    test_model(m)
    plot_results(losses, accuracies)