from const import *
import cnn
import video
import vis

import datetime

# =============================================================================
# TODO:
#     - data punten vervangen door plaatjes (achtergrondkleur = class)
#     - andere lagen visualiseren
#     - video opties aanpassen
#     - (draaien op cluster)
#     - Fitten op ALLE data, dan pas per frame visualizeren
#     - Per batch visualizeren
#     - Seed van UMAP hetzelfde?
#     - Learning rate omlaag voor meer epochs
#     - Batch normalization
# 
# =============================================================================


if __name__ == '__main__':
    print("Hello and welcome to the CNN training visualizer.")
    start_time = datetime.datetime.now();
    output_dir = f"output/{start_time.strftime('%Y-%m-%d %H.%M.%S')}"
    print(f"Started {start_time}")

    m = cnn.ConvNet()
    classes, train_loader, losses, accuracies = cnn.train_model(m, output_dir)
    vis.show_images(train_loader, classes)
    # vis.plot_results(losses, accuracies)
    # Load a model instead: m = torch.load(MODEL_STORE_PATH + 'conv_net_model.ckpt')
    cnn.test_model(m)
    video.make(output_dir)
