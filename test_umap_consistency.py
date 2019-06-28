from const import *
import matplotlib.pyplot as plt;
from matplotlib.lines import Line2D
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.gridspec as gridspec
import matplotlib as mpl
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
from PIL import Image
import random

start_time = datetime.datetime.now();
output_dir = f"output/umap not shuffled {start_time.strftime('%Y-%m-%d %H.%M.%S')} {train_dir.split('/')[-1]} {test_dir.split('/')[-1]}"

def test_umap(n):
    
    alpha = 0.8
    colours = [[0, 0, 1], [1, 0.2, 0.2], [0.3, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0.8, 0], [1, 0.9, 0.9], [0.6, 0.3, 1], [0.4, 1, 0.7], [1, 0.4, 0]]
    colours_long = copy.deepcopy(colours)
    for col in colours_long:
        col.append(1)
    newcmp = ListedColormap(colours_long)
    colours_alpha = copy.deepcopy(colours)
    for col in colours_alpha:
        col.append((1+alpha)/2)
    barcmp = ListedColormap(colours_alpha)
    
    
    VIS_DATA = np.loadtxt("train.txt")
    TEST_DATA = np.loadtxt("test.txt")
    TEST_PRED = np.loadtxt("test_c.txt")
    VIS_PRED = np.loadtxt("train_c.txt")
    '''
    VIS_ALL = list(zip(VIS_DATA, VIS_PRED))
    random.shuffle(VIS_ALL)
    VIS_DATA, VIS_PRED = zip(*VIS_ALL)
    '''
    fit = umap.UMAP(random_state=np.random.seed(42), n_neighbors=n_neigh, transform_seed=np.random.seed(42))
    all_data = np.concatenate((VIS_DATA, TEST_DATA), axis=0)
    trans = fit.fit(all_data)
    train_emb = trans.transform(VIS_DATA)
    test_emb = trans.transform(TEST_DATA)
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
    ax.set_title("Images for testing", fontsize=24)
    ax.scatter(x, y, c=TEST_PRED, s=10, cmap=newcmp)
    #plt.setp(ax, xticks=[], yticks=[])
    sm = plt.cm.ScalarMappable(cmap=barcmp)
    sm._A = []
    cb = plt.colorbar(sm)
    cb.ax.set_title('Predicted class', fontsize=24)
    #cb.set_ticks(np.arange(0.05,1.05,step=0.1))
    #cb.set_ticklabels(classes)
    #cb.ax.tick_params(labelsize=24)
    
    # set up sub plot 2 (train data)
    ax2 = plt.subplot(big_x[0,3])
    ax2.set_title("Images for training", fontsize=24)
    ax2.scatter(train_x, train_y, c=VIS_PRED, s=1, cmap=newcmp)
    #plt.setp(ax2, xticks=[], yticks=[])
    '''
    # set up sub plot 3 (accuracy)
    ax4 = plt.subplot(big_x[1,3])
    ax4.set_title("Accuracy of the neural network", fontsize=24)
    ax4.plot(*zip(*self.VIS_ACC), label='train set')
    ax4.plot(*zip(*self.TEST_ACC), label='test set')
    ax4.legend(loc='lower right', fontsize=24)
    plt.xticks(np.arange(0,num_epochs+1,step=2))
    plt.yticks(np.arange(0,1.2,step=0.2))
    ax4.tick_params(labelsize=22)
    '''
    # set up figure information
    fig.set_size_inches(w=28, h=16)
    plt.suptitle(f"The neural network's activity when given sketch images, after training for 3 epochs", fontsize=30)
    plt.figtext(.5,.93,"Activations of all 100 neurons in the CNN's first fully connected layer, reduced to 2 dimensions using UMAP", fontsize=24, ha='center')
    
    # set up output information
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    fig.savefig(f"{output_dir}/{n} after epoch 3 of 20 (#c={num_classes}, bs={batch_size}, lr={learning_rate}).png")
    
for i in range(10):
    test_umap(i)
    i+=1