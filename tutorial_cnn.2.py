import torch;
import torchvision;
from torchvision import datasets;
from torch.utils.data import Dataset, random_split;
import torchvision.transforms as transforms;
import matplotlib.pyplot as plt;
import numpy as np;

# Sketch data
sketch_dir =  r"./data/train"
trainset = datasets.ImageFolder(
        sketch_dir,
        transform=transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize([256, 256]),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                #std=[0.229, 0.224, 0.225])
        ])
    )
classes = trainset.classes
# Sample data
# size = len(trainset.samples);
# train_size = 15000
# test_size = size - train_size
# trainset, testset = random_split(trainset, (train_size, test_size));
# Load data
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True
)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=4, shuffle=True
# )


# Show random images
def show_images():
    # Get some random training images
    dataiter = iter(trainloader);
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

if __name__ == '__main__':
    show_images();
