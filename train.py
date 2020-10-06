import numpy as np
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import ReLU
from keras.utils import plot_model
from utils import define_model, prepare_dataset


def train(iteration=3,
          DATASET='DRIVE',
          TRANSFER_LRN_DATASET=None,
          crop_size=128,
          need_au=True,
          ACTIVATION='ReLU',
          dropout=0.1,
          lr=1e-3,
          batch_size=32,
          repeat=4,
          minimum_kernel=32,
          epochs=200,
          pretrained_model=None,
          suffix='',
          gpus=1):
    model_name = f"Final_Emer_Iteration_{iteration}_cropsize_{crop_size}_lr_{lr}_epochs_{epochs}{suffix}"

    if TRANSFER_LRN_DATASET is None:
        TRANSFER_LRN_DATASET = DATASET

    print("> Training model : %s" % model_name)
    if pretrained_model is not None:
        print('> Using transfer learning from dataset %s' % TRANSFER_LRN_DATASET)

    prepare_dataset.prepareDataset(DATASET)

    activation = globals()[ACTIVATION]
    model = define_model.get_unet(minimum_kernel=minimum_kernel, do=dropout, lr=lr,
                                  activation=activation, iteration=iteration, gpus=gpus, pretrained_model=pretrained_model)

    try:
        os.makedirs(f"trained_model/{DATASET}-{TRANSFER_LRN_DATASET}/", exist_ok=True)
        os.makedirs(f"logs/{DATASET}-{TRANSFER_LRN_DATASET}/", exist_ok=True)
    except:
        pass

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d---%H-%M-%S")

    tensorboard = TensorBoard(
        log_dir=f"logs/{DATASET}-{TRANSFER_LRN_DATASET}/Final_Emer-Iteration_{iteration}-Cropsize_{crop_size}-LR_{lr}-Epochs_{epochs}{suffix}---{date_time}",
        histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
        write_images=True, embeddings_freq=0, embeddings_layer_names=None,
        embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    save_path = f"trained_model/{DATASET}-{TRANSFER_LRN_DATASET}/{model_name}.hdf5"
    checkpoint = ModelCheckpoint(save_path, monitor='final_out_loss', verbose=1, save_best_only=True, mode='min')

    data_generator = define_model.Generator(batch_size, repeat, DATASET)

    model.summary()
    history = model.fit_generator(data_generator.gen(au=need_au, crop_size=crop_size, iteration=iteration),
                                  epochs=epochs, verbose=1,
                                  steps_per_epoch=100 * data_generator.n // batch_size,
                                  use_multiprocessing=True, workers=8,
                                  callbacks=[tensorboard, checkpoint])


if __name__ == "__main__":

    dataset = 'DROPS'
    transfer_lrn_dataset = 'UNIMODEL'
    pretrained_model = f'./pretrained/{transfer_lrn_dataset}/weights.hdf5'
    gpus=2 #For use on cluster only. 

    # transfer_lrn_dataset = None
    # pretrained_model = None

    print(f"> Training on {dataset} dataset.")

    train(
        iteration=3,
        DATASET=dataset, # DRIVE, CHASEDB1, STARE, HRF, DROPS
        TRANSFER_LRN_DATASET=transfer_lrn_dataset,
        pretrained_model=pretrained_model,
        batch_size=32,
        lr=1e-12,
        crop_size=128,
        epochs=10,
        suffix='unet-conv1-finetune',
        need_au=True,
        gpus=gpus
    )
