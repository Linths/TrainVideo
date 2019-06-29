# Visualizing the training process of a convoluted neural network over time.
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


# Hyperparameters, general NN settings
num_epochs = 20
num_classes = 10
batch_size = 25
learning_rate = 0.0001

# Data access
train_dir =  r"./data/train_sub_trans"
test_dir =  r"./data/test_sub_2"
model_dir = r"./model"

# NN scaling params
image_width = 128       # Image width / height
no_dimens = 50          # Number of dimensions that will be reduced by UMAP to 2. Size of the 2nd fully connected layer.

# Visualization
show_after_epochs = 1
print_every_vis = False
n_neigh = 25
