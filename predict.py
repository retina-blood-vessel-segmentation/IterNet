############Test


import os
import tensorflow as tf
from keras.backend import tensorflow_backend

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

from utils import define_model, tools
from utils.evaluate import evaluate
from keras.layers import ReLU
from tqdm import tqdm
import numpy as np
from skimage.transform import resize
import cv2
from sklearn.metrics import roc_auc_score
import math
import pickle
import click
import mlflow
from keras.callbacks import ModelCheckpoint

@click.command()
@click.option('--activation', default="ReLU")
@click.option('--dropout', type=click.FLOAT, default=0.1)
@click.option('--batch_size', type=click.INT, default=32)
@click.option('--minimum_kernel', type=click.INT, default=32)
@click.option('--iteration', type=click.INT, default=3)
@click.option('--lr', type=click.FLOAT, default=1e-3)
@click.option('--model_path', help='Path to the h5 file with weights used for prediction.')
@click.option('--test_images_dir', help='A directory containing images for prediction.')
@click.option('--test_labels_dir', help='A directory containing corresponding groundtruth images for prediction images.')
@click.option('--test_masks_dir', help='A directory containing corresponding masks for prediction images.')
@click.option('--dataset', type=click.Choice(['DRIVE', 'STARE', 'CHASE', 'DROPS','DRIVE-eval','STARE-eval','CHASE-eval']),
              help='A case-sensitive dataset name that will be used for inference. ')
@click.option('--output_dir', help='Path to the directory where inference results will be saved.')
def predict(activation, dropout, batch_size, minimum_kernel, iteration, lr, model_path, test_images_dir, test_labels_dir, test_masks_dir, dataset, output_dir):
    with mlflow.start_run() as run:
        mlflow.log_params({
            'dataset': dataset,
            'model_path': model_path,
            'lr': lr,
            'dropout' : dropout,
            'minimum_kernel': minimum_kernel,
            'iteration' : iteration,
            'batch_size' : batch_size
        })
        print(f'> Predicting on {dataset} dataset.')
        size = tools.get_desired_size(dataset)

        act = globals()[activation]
        model = define_model.get_unet(minimum_kernel=minimum_kernel, do=dropout, activation=act, iteration=iteration, pretrained_model=model_path)
        probability_maps_dir = os.path.join(output_dir, "probability_maps")
        segmentation_masks_dir = os.path.join(output_dir, "segmentation_masks")
        if not os.path.exists(probability_maps_dir):
            os.makedirs(probability_maps_dir)
        if not os.path.exists(segmentation_masks_dir):
            os.makedirs(segmentation_masks_dir)
        print(f'> Probability maps will be saved at {os.path.abspath(probability_maps_dir)}')
        print(f'> Segmentation masks will be saved at {os.path.abspath(segmentation_masks_dir)}')
        flag = True
        bucket = 50
        startat = 0
        outw = 0
        outh = 0
        if dataset.startswith("DRIVE"):
            outw = 565
            outh = 584
        elif dataset.startswith("CHASE"):
            outw = 999
            outh = 960
        elif dataset.startswith("STARE"):
            outw = 700
            outh = 605
        elif dataset.startswith("DROPS"):
            outw = 640
            outh = 480
        else:
            print("Fatal error, unknown dataset.")
            exit(1)
        while flag:
            x_test, _, _, fnames = tools.load_files(test_images_dir, test_labels_dir, size, tools.get_label_pattern_for_dataset(dataset),
                                        mode='test',startat=startat,bucket=bucket)
            if len(x_test) == 0:
                flag = False
                continue
            for i, x in enumerate(x_test):                
                y = model.predict(np.array([x]))[0]
                y = tools.crop_to_shape(y, (0, outh, outw, 1))
                y_ = y[0, :, :, 0]
                cv2.imwrite(os.path.join(probability_maps_dir, f"{fnames[i]}.png"), y_ * 255)
            startat = startat + bucket

if __name__ == "__main__":
    predict() # pylint: disable=no-value-for-parameter