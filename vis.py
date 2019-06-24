from const import *

import matplotlib.pyplot as plt;
from matplotlib.lines import Line2D
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.gridspec as gridspec
import seaborn as sns
import umap
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import torchvision
import numpy as np;
import os
import copy
from skimage.transform import resize
from skimage import data, color, io, img_as_float
import torch
import torch.nn as nn
from torch.nn import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

class Visualization():
    alpha = 0.6
    colours = [[0, 0, 1], [1, 0, 0], [0.3, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0.8, 0], [1, 0.9, 0.9], [0.6, 0.3, 1], [0.4, 1, 0.7], [1, 0.4, 0]]
    # colours: blue, red, green, magenta, cyan, yellow, salmon, violet, turquoise, orange
    
    # class instances below will be initalized in init function
    newcmp = None
    barcmp = None
    plot_image = []

    def __init__(self, classes, VIS_DATA=[], VIS_TARGET=[], VIS_PRED=[], TEST_DATA=[], TEST_TARGET=[], TEST_PRED=[], VIS_ACC=[], TEST_ACC=[]):
        self.classes = classes
        self.VIS_DATA = VIS_DATA
        self.VIS_TARGET = VIS_TARGET
        self.VIS_PRED = VIS_PRED
        self.TEST_DATA = TEST_DATA
        self.TEST_TARGET = TEST_TARGET
        self.TEST_PRED = TEST_PRED
        self.VIS_ACC = VIS_ACC
        self.TEST_ACC = TEST_ACC
        
        Visualization.plot_image = []
        
        # make two RGBA versions of the colour list and turn them into colormaps
        colours_long = copy.deepcopy(Visualization.colours)
        for col in colours_long:
            col.append(1)
        Visualization.newcmp = ListedColormap(colours_long)
        colours_alpha = copy.deepcopy(Visualization.colours)
        for col in colours_alpha:
            col.append((1+Visualization.alpha)/2)
        Visualization.barcmp = ListedColormap(colours_alpha)

        # start the accuracy plots at (0,0)
        self.VIS_ACC.append((0,0))
        self.TEST_ACC.append((0,0))

    def make_label_vis(self, output_dir, epochs_passed):
        # make a visualisation of the neural network's training process with 3 subplots.
        # 1. a visualisation of the test data with images as plot points and a colorbar as legend
        # 2. a visualisation of the training data
        # 3. a visualisation of the accuracy of the test and training data

        # set up umap dimensionality reduction
        fit = umap.UMAP(random_state=np.random.seed(42), n_neighbors=n_neigh)
        all_data = np.concatenate((self.VIS_DATA, self.TEST_DATA), axis=0)
        trans = fit.fit(all_data)
        train_emb = trans.transform(self.VIS_DATA)
        test_emb = trans.transform(self.TEST_DATA)
        x = test_emb[:,0]
        y = test_emb[:,1]
        train_x = train_emb[:,0]
        train_y = train_emb[:,1]

        # set up figure
        fig = plt.figure(1)
        big_x = gridspec.GridSpec(2,4)
        big_x.update(wspace=0.3, hspace=0.5)

        # set up sub plot 1 (test data)
        ax = plt.subplot(big_x[:,:3])
        ax.title.set_text("Images for testing")
        ax.scatter(x, y, c=self.TEST_PRED, s=1, cmap=Visualization.newcmp)
        plt.setp(ax, xticks=[], yticks=[])
        for x0, y0, img in zip(x, y, Visualization.plot_image):
            ab = AnnotationBbox(OffsetImage(img), (x0, y0), frameon=False)
            ax.add_artist(ab)
        sm = plt.cm.ScalarMappable(cmap=Visualization.barcmp)
        sm._A = []
        cb = plt.colorbar(sm)
        cb.ax.set_title('Predicted class')
        cb.set_ticks(np.arange(0.05,1.05,step=0.1))
        cb.set_ticklabels(self.classes)

        # set up sub plot 2 (train data)
        ax2 = plt.subplot(big_x[0,3])
        ax2.title.set_text("Images for training")
        ax2.scatter(train_x, train_y, c=self.VIS_PRED, s=1, cmap=Visualization.newcmp)
        plt.setp(ax2, xticks=[], yticks=[])

        # set up sub plot 3 (accuracy)
        ax4 = plt.subplot(big_x[1,3])
        ax4.title.set_text("Accuracy of the neural network")
        ax4.plot(*zip(*self.VIS_ACC), label='train set')
        ax4.plot(*zip(*self.TEST_ACC), label='test set')
        ax4.legend(loc='lower right')
        plt.xticks(np.arange(0,num_epochs+1,step=2))
        plt.yticks(np.arange(0,1.1,step=0.1))

        # set up figure information
        fig.set_size_inches(w=28, h=16)
        plt.suptitle(f"The neural network's activaty when given sketch images, after training for {epochs_passed} epochs", fontsize=18)
        plt.figtext(.5,.95,"Activations of all 100 neurons in the CNN's first fully connected layer, reduced to 2 dimensions using UMAP", fontsize=14, ha='center')
        
        # set up output information
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        fig.savefig(f"{output_dir}/after epoch {epochs_passed:02d} of {num_epochs} (#c={num_classes}, bs={batch_size}, lr={learning_rate}).png")

        # checking if it should print the figure
        if print_every_vis:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

    def add_class_colour(self, image, label):
        # load the image and add a mask in the colour of the corresponding label, with transparency alpha.
        # then resize the image to make it an appropriate plot point size and append it to the image list.
        img = img_as_float(image)[0]
        rows, cols = img.shape
        color_mask = np.zeros((rows, cols, 3))
        color_mask[0:rows,0:cols] = Visualization.colours[label]
        img_color = np.dstack((img,img,img))
        img_hsv = color.rgb2hsv(img_color)
        color_mask_hsv = color.rgb2hsv(color_mask)
        img_hsv[...,0]=color_mask_hsv[...,0]
        img_hsv[...,1]=color_mask_hsv[...,1]*Visualization.alpha
        img_masked = color.hsv2rgb(img_hsv)
        img_small = resize(img_masked, (40,40), anti_aliasing=True, mode='constant')
        img_small.clip(min=0, out=img_small)
        Visualization.plot_image.append(img_small)

    def clear_after_epoch(self):
        # clear all per-epoch-data containing lists
        self.VIS_DATA.clear()
        self.VIS_TARGET.clear()
        self.VIS_PRED.clear()
        self.TEST_DATA.clear()
        self.TEST_TARGET.clear()
        self.TEST_PRED.clear()
        Visualization.plot_image.clear()

def show_images(train_loader, classes):
    # get some random images of the train set, show them and print first four labels
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

def imshow(img):
    # show an image
    img = img / 2 + 0.5;    # unnormalize
    npimg = img.numpy();
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def plot_results(loss_list, acc_list):
    # make a plot of the train accuracy and loss
    p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title=f'Performance of sketches CNN [#classes = {num_classes}, batch size = {batch_size}, #epochs = {num_epochs}, learning rate = {learning_rate}]')
    p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
    p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
    p.line(np.arange(len(loss_list)), loss_list)
    p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
    show(p)
