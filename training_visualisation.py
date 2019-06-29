from const import *
import cnn
#import video
import vis

import datetime
import shutil
import os

# The sketch dataset is licensed under a Creative Commons Attribution 4.0 International License.
# https://creativecommons.org/licenses/by/4.0/
# Copyright (C) 2012 Mathias Eitz, James Hays, and Marc Alexa. 2012. How Do Humans Sketch Objects? ACM Trans. Graph. (Proc. SIGGRAPH) 31, 4 (2012), 44:1--44:10.
# http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/
# The data has been modified from original by splitting the images in train sets and test sets, and subsets

if __name__ == '__main__':
    print("Hello and welcome to the CNN training visualizer.")
    start_time = datetime.datetime.now();
    output_dir = f"output/{start_time.strftime('%Y-%m-%d %H.%M.%S')} {train_dir.split('/')[-1]} {test_dir.split('/')[-1]}"
    print(f"Started {start_time}\n")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    shutil.copyfile("const.py", output_dir + "/parameters.py")

    m = cnn.ConvNet()
    cnn.train_model(m, output_dir)
    # classes, train_loader, losses, accuracies = cnn.train_model(m, output_dir)
    # vis.show_images(train_loader, classes)
    # vis.plot_results(losses, accuracies)
    # Load a model instead: m = torch.load(MODEL_STORE_PATH + 'conv_net_model.ckpt')
    # cnn.test_model(m)
    # video.make(output_dir)
