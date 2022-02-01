import argparse
from datasets import create_dataset
from utils import parse_configuration
from models import create_model
import os
from utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import numpy as np

"""Performs validation of a specified model.

Input params:
    config_file: Either a string with the path to the JSON
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and
        model-specific settings.
"""
def validate(configuration):

    print('Initializing dataset...')
    val_dataset = create_dataset(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print('The number of validation samples = {0}'.format(val_dataset_size))

    print('Initializing model...')
    model = create_model(configuration['model_params'])
    model.setup()
    model.eval()

    print('Initializing visualization...')
    visualizer = Visualizer(configuration['visualization_params_validation'])   # create a visualizer that displays images and plots

    model.pre_epoch_callback(configuration['model_params']['load_checkpoint'])

    #Loops through all validation data and runs though model
    for i, data in enumerate(val_dataset):
        if (i > 0):
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        img = np.zeros((512, 512, 3))
        out_img = np.zeros((512, 512, 3))
        for i in range(512):
            for j in range(512):
                for k in range(3):
                    img[i,j,k] = float(data[0][0][k][i][j])
                    out_img[i,j,k] = float(model.output[0][k][i][j])
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img)
        axs[1].imshow(out_img)
        # plt.savefig("./plots/epoch_{}.png".format(configuration['model_params']['load_checkpoint']))
        plt.show()

    #Where results are calculated and visualized
    # model.post_epoch_callback(configuration['model_params']['load_checkpoint'], visualizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()

    print('Reading config file...')
    configuration = parse_configuration(args.configfile)
    if (configuration['model_params']['load_checkpoint'] == -2):
        for epoch in range(configuration['model_params']['epoch_list'][0], configuration['model_params']['epoch_list'][1]):
            configuration['model_params']['load_checkpoint'] = epoch
            validate(configuration)
    else:
        validate(configuration)
