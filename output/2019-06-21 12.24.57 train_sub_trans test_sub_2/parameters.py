# Hyperparameters, general NN settings
num_epochs = 12
num_classes = 10
batch_size = 50
learning_rate = 0.00005

# Data access
train_dir =  r"./data/train_sub_trans"
test_dir =  r"./data/test_sub_2"
model_dir = r"./model"

# NN scaling params
image_width = 48       # Image width / height
no_dimens = 100         # Number of dimensions that will be reduced by UMAP to 2. Size of the 2nd fully connected layer.

# Visualization
show_after_epochs = 1
print_every_vis = False
n_neigh = 25