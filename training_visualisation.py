from const import *
import cnn
#import video
import vis

import datetime
import shutil
import os

# =============================================================================
# TODO:
#     - andere lagen visualiseren
#     - Per batch visualizeren
#     - Batch normalization
# 
# =============================================================================


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
