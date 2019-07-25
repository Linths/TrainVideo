# Visualizing the training process of a convolutional neural network over time.
# Copyright (C) 2019  Michelle Peters & Lindsay Kempen

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
import os
from const import *

def make(output_dir):
    # Uncomment for manual test output_dir = "output/testimages"; # "output/2019-05-28 11.11.28";
    print(output_dir)
    images = []
    for filename in os.listdir(output_dir):
        if (filename.endswith(".png")):
            print(filename)
            img = cv2.imread(output_dir + "/" + filename)
            height, width, layers = img.shape
            size = (width,height)
            images.append(img)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video_dir = f"{output_dir}/total {num_epochs} passed (#c={num_classes}, bs={batch_size}, lr={learning_rate}).avi"
    fps = 0.25
    out = cv2.VideoWriter(video_dir, fourcc, fps, size)
    for i in range(len(images)):
        out.write(images[i])
    out.release()

def makeFromLast():
    all_outputs = [folder for folder in  os.listdir("output") if folder.startswith('2019')]
    all_outputs.sort()
    return make("output/" + all_outputs[-1])

if __name__ == '__main__':
    makeFromLast()