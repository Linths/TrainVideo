import cv2
import os
from const import *

def make(output_dir):
    # Uncomment for manual test output_dir = "output/testimages"; # "output/2019-05-28 11.11.28";
    print(output_dir)
    images = []
    for filename in os.listdir(output_dir):
        if not (filename.endswith(".avi")):
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
        print(f"written image {i}")
        out.write(images[i])
    out.release()

def makeFromLast():
    all_outputs = [folder for folder in  os.listdir("output") if folder.startswith('2019')]
    all_outputs.sort()
    return make("output/" + all_outputs[-1])

if __name__ == '__main__':
    makeFromLast()