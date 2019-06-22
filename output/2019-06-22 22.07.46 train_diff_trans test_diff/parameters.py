# Hyperparameters, general NN settings
num_epochs = 20
num_classes = 10
batch_size = 25
learning_rate = 0.00001

# Data access
train_dir =  r"./data/train_diff_trans"
test_dir =  r"./data/test_diff"
model_dir = r"./model"

# NN scaling params
image_width = 128       # Image width / height
no_dimens = 50          # Number of dimensions that will be reduced by UMAP to 2. Size of the 2nd fully connected layer.

# Visualization
show_after_epochs = 1
print_every_vis = False
n_neigh = 25