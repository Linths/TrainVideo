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
    colours = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0.8, 0], [1, 0.9, 0.9], [0.6, 0.3, 1], [0.4, 1, 0.9], [1, 0.5, 0]]
    # blue, red, green, pink, cyan, yellow, white, violet, teal, orange
    # Class instances below will be initalized in init function
    newcmp = None
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
        colours_long = copy.deepcopy(Visualization.colours)
        for col in colours_long:
            col.append(1)
        Visualization.newcmp = ListedColormap(colours_long)
        
        
        Visualization.plot_image = []
        self.VIS_ACC.append((0,0))
        self.TEST_ACC.append((0,0))

    def make_vis(self, output_dir, epochs_passed):
        sns.set(style='white', rc={'figure.figsize':(10,8)})
        cmap=plt.cm.hsv  #Spectral #misschien .hsv (regenboog)
        new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0.0, b=0.9), cmap(np.linspace(0.0,0.9,10)))
        
        markers = ['$0$','$1$','$2$','$3$','$4$','$5$','$6$','$7$','$8$','$9$']
        fit = umap.UMAP(random_state=np.random.seed(42), n_neighbors=n_neigh)
        standard_embedding = fit.fit_transform(self.VIS_DATA)
        test_embedding = fit.fit_transform(self.TEST_DATA)
        
        #different marker style per actual class
        #training set
        
        # MAKE SCALABLE TO AMOUNT OF CLASSES-----------------------------------
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 = [],[],[],[],[],[],[],[],[],[]
        for i in range(len(standard_embedding)):
            if self.VIS_TARGET[i]==0:
                t0.append([standard_embedding[i,0], standard_embedding[i,1], self.VIS_PRED[i], self.VIS_TARGET[i]])
            if self.VIS_TARGET[i]==1:
                t1.append([standard_embedding[i,0], standard_embedding[i,1], self.VIS_PRED[i], self.VIS_TARGET[i]])
            if self.VIS_TARGET[i]==2:
                t2.append([standard_embedding[i,0], standard_embedding[i,1], self.VIS_PRED[i], self.VIS_TARGET[i]])
            if self.VIS_TARGET[i]==3:
                t3.append([standard_embedding[i,0], standard_embedding[i,1], self.VIS_PRED[i], self.VIS_TARGET[i]])
            if self.VIS_TARGET[i]==4:
                t4.append([standard_embedding[i,0], standard_embedding[i,1], self.VIS_PRED[i], self.VIS_TARGET[i]])
            if self.VIS_TARGET[i]==5:
                t5.append([standard_embedding[i,0], standard_embedding[i,1], self.VIS_PRED[i], self.VIS_TARGET[i]])
            if self.VIS_TARGET[i]==6:
                t6.append([standard_embedding[i,0], standard_embedding[i,1], self.VIS_PRED[i], self.VIS_TARGET[i]])
            if self.VIS_TARGET[i]==7:
                t7.append([standard_embedding[i,0], standard_embedding[i,1], self.VIS_PRED[i], self.VIS_TARGET[i]])
            if self.VIS_TARGET[i]==8:
                t8.append([standard_embedding[i,0], standard_embedding[i,1], self.VIS_PRED[i], self.VIS_TARGET[i]])
            if self.VIS_TARGET[i]==9:
                t9.append([standard_embedding[i,0], standard_embedding[i,1], self.VIS_PRED[i], self.VIS_TARGET[i]])
        
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 = np.array(t0), np.array(t1), np.array(t2), np.array(t3), np.array(t4), np.array(t5), np.array(t6), np.array(t7), np.array(t8), np.array(t9)
        
        x = 1/(len(self.classes)-1)
        
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
        plt.scatter(t0[:, 0], t0[:, 1], s=50, color=new_cmap(x*t0[:,2]+0.01), marker=markers[0])
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
            if self.TEST_TARGET[i]==0:
                v0.append([test_embedding[i,0], test_embedding[i,1], self.TEST_PRED[i], self.TEST_TARGET[i]])
            if self.TEST_TARGET[i]==1:
                v1.append([test_embedding[i,0], test_embedding[i,1], self.TEST_PRED[i], self.TEST_TARGET[i]])
            if self.TEST_TARGET[i]==2:
                v2.append([test_embedding[i,0], test_embedding[i,1], self.TEST_PRED[i], self.TEST_TARGET[i]])
            if self.TEST_TARGET[i]==3:
                v3.append([test_embedding[i,0], test_embedding[i,1], self.TEST_PRED[i], self.TEST_TARGET[i]])
            if self.TEST_TARGET[i]==4:
                v4.append([test_embedding[i,0], test_embedding[i,1], self.TEST_PRED[i], self.TEST_TARGET[i]])
            if self.TEST_TARGET[i]==5:
                v5.append([test_embedding[i,0], test_embedding[i,1], self.TEST_PRED[i], self.TEST_TARGET[i]])
            if self.TEST_TARGET[i]==6:
                v6.append([test_embedding[i,0], test_embedding[i,1], self.TEST_PRED[i], self.TEST_TARGET[i]])
            if self.TEST_TARGET[i]==7:
                v7.append([test_embedding[i,0], test_embedding[i,1], self.TEST_PRED[i], self.TEST_TARGET[i]])
            if self.TEST_TARGET[i]==8:
                v8.append([test_embedding[i,0], test_embedding[i,1], self.TEST_PRED[i], self.TEST_TARGET[i]])
            if self.TEST_TARGET[i]==9:
                v9.append([test_embedding[i,0], test_embedding[i,1], self.TEST_PRED[i], self.TEST_TARGET[i]])
        
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
        for j in range(len(self.classes)):
            legend_colors.append(Line2D([0], [0], marker=markers[j], color='w', label=self.classes[j],
                    markerfacecolor=new_cmap(j/9), markersize=10))
        ax.legend(handles=legend_colors, loc='center')
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_locator(plt.NullLocator())

        plt.subplot(2,3,6)
        plt.title(f"accuracy")
        plt.plot(*zip(*self.VIS_ACC), label='train set')
        plt.plot(*zip(*self.TEST_ACC), label='test set')
        plt.legend(loc='lower right')
        
        plt.suptitle(f"Neuron activations of the sketches CNN after {epochs_passed} epochs")
        
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        plt.savefig(f"{output_dir}/after epoch {epochs_passed:02d} of {num_epochs} (#c={num_classes}, bs={batch_size}, lr={learning_rate}).png")
        
        if print_every_vis:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

    def make_label_vis(self, output_dir, epochs_passed):
        #make test vis with images
        def getImage(image):
            return OffsetImage(image)
        
        #setup umap dimensionality reduction
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

        # big plot (test data)
        ax = plt.subplot(big_x[:,:3])
        ax.title.set_text("test data")
        ax.scatter(x, y, c=self.TEST_PRED, s=1, cmap=Visualization.newcmp)
        plt.setp(ax, xticks=[], yticks=[])
        for x0, y0, img in zip(x, y, Visualization.plot_image):
            ab = AnnotationBbox(getImage(img), (x0, y0), frameon=False)
            ax.add_artist(ab)
        sm = plt.cm.ScalarMappable(cmap=Visualization.newcmp)
        sm._A = []
        cb = plt.colorbar(sm)
        cb.set_ticks(np.arange(0.05,1.05,step=0.1))
        cb.set_ticklabels(self.classes)

        # sub plot 2 (train data)
        ax2 = plt.subplot(big_x[0,3])
        ax2.title.set_text("train predicted")
        ax2.scatter(train_x, train_y, c=self.VIS_PRED, s=1, cmap=Visualization.newcmp)
        plt.setp(ax2, xticks=[], yticks=[])
        
        # sub plot 3 (accuracy)
        ax4 = plt.subplot(big_x[1,3])
        ax4.set_title("accuracy")
        ax4.plot(*zip(*self.VIS_ACC), label='train set')
        ax4.plot(*zip(*self.TEST_ACC), label='test set')
        ax4.legend(loc='lower right')
        plt.xticks(np.arange(0,num_epochs+1,step=1))
        plt.yticks(np.arange(0,1.1,step=0.1))
        
        # general info
        fig.set_size_inches(w=14, h=8)
        plt.suptitle(f"Neuron activations of the sketches CNN after {epochs_passed} epochs")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        fig.savefig(f"{output_dir}/after epoch {epochs_passed:02d} of {num_epochs} (#c={num_classes}, bs={batch_size}, lr={learning_rate}).png")

        if print_every_vis:
            plt.show()
        else:
            plt.clf()
            plt.cla()
            plt.close()

    def add_class_colour(self, image, label):
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
        img_small = resize(img_masked, (20,20), anti_aliasing=True, mode='constant')
        img_small.clip(min=0, out=img_small)
        # img_small[0][0][0]=value
        Visualization.plot_image.append(img_small)

    def clear_after_epoch(self):
        self.VIS_DATA.clear()
        self.VIS_TARGET.clear()
        self.VIS_PRED.clear()
        self.TEST_DATA.clear()
        self.TEST_TARGET.clear()
        self.TEST_PRED.clear()
        Visualization.plot_image.clear()

# Show random images
def show_images(train_loader, classes):
    # Get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    # Show images
    imshow(torchvision.utils.make_grid(images))
    # Print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

def imshow(img):
    img = img / 2 + 0.5;    # Unnormalize
    npimg = img.numpy();
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def plot_results(loss_list, acc_list):
    p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title=f'Performance of sketches CNN [#classes = {num_classes}, batch size = {batch_size}, #epochs = {num_epochs}, learning rate = {learning_rate}]')
    p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
    p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
    p.line(np.arange(len(loss_list)), loss_list)
    p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
    show(p)