# TrainGif

_Visualizing the training process of a convoluted neural network over time._

This tool is extensively discussed in our paper "Neural Networks for Non-Experts: Intuitively Visualising the Training Process Over Time".

## Copyright

### Data set
We trained our CNN on the sketch dataset. The sketch dataset is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). Copyright (C) [2012 Mathias Eitz, James Hays, and Marc Alexa. 2012. How Do Humans Sketch Objects? ACM Trans. Graph. (Proc. SIGGRAPH) 31, 4 (2012), 44:1--44:10](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/). The data has been modified from original by splitting the images in train sets and test sets, and subsets.

### This program
Visualizing the training process of a convoluted neural network over time.  
Copyright (C) 2019  Michelle Peters & Lindsay Kempen

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by 
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.  
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.  
You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

## Run instructions
Python 3.7 is advised. Several libraries are necessary, such as Pytorch (neural network) and OpenCV (video making).

### Run it
1. Run `python training_visualisation.py` to create visualisations on pre-specified parameters
2. Visualisation frames are in the `output` folder.
3. To make a video from the frames, run `python video.py`

### Tweak it
- Tweak the program parameters in [`const.py`](const.py) where desired. It contains CNN parameters, visualization parameters, and more. 
- Change the image classes?
    1. Create a new training subset and testing subset in the `data` folder
    2. If desired, you can transform the training images using `transform_images.py`
    3. Update `num_classes`, `train_dir` and `test_dir` in [`const.py`](const.py)
