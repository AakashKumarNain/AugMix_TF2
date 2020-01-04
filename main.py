import os
import argparse
import logging
import numpy as np
import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.keras.utils import to_categorical
from trainer import train

###########################################################################


def get_cifar_data(num_classes=10):
    """Loads cifar-10 data. Normalize the images and do one-hot encoding for labels"""

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.

    y_train_cat = to_categorical(y_train, num_classes=num_classes).astype(np.float32)
    y_test_cat = to_categorical(y_test, num_classes=num_classes).astype(np.float32)

    return x_train, y_train, x_test, y_test, y_train_cat, y_test_cat


###########################################################################

def parse_args():
    # Parse input arguments
    desc = "Implementation for AugMix paper"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--batch_size',
                        type=int,
                        required=False)
    parser.add_argument('--epochs',
                        help='number of training epochs',
                        type=int,
                        required=True)
    parser.add_argument('--max_lr',
                        help='maxium learning rate for lr scheduler',
                        default=1.0,
                        type=float)
    parser.add_argument('--min_lr',
                        help='minimum learning rate for lr scheduler',
                        default=1e-5,
                        type=float)
    parser.add_argument('--img_size',
                        help='size of the images',
                        default=32,
                        type=int)
    parser.add_argument("--save_dir_path",
                        type=str,
                        help="dir path to save output results",
                        default="",
                        required=False)
    parser.add_argument("--plot_name",
                        type=str,
                        help="filename for the plots",
                        default="history.png",
                        required=False)
    args = vars(parser.parse_args())
    return args

###########################################################################

def main():
    args = parse_args()
    print('\nCalled with args:')
    for key in args:
        print(f"{key:<10}:  {args[key]}")
    print("="*78)

    # get the command line args
    config.max_lr = args["max_lr"]
    config.min_lr = args["min_lr"]
    config.batch_size = args["batch_size"]
    config.num_epochs = args["epochs"]
    config.IMAGE_SIZE = args["img_size"]
    config.plt_name = args["plot_name"]
    
    if args["save_dir_path"] == "":
        config.save_dir_path = './model_checkpoints'
    else:
        config.save_dir_path = args["save_dir_path"]
        


    # get the data
    print("\nLoading data now.", end=" ")
    x_train, y_train, x_test, y_test, y_train_cat, y_test_cat = get_cifar_data()
    training_data = [x_train, y_train, y_train_cat]
    validation_data = [x_test, y_test, y_test_cat]
    print("Data loading complete. \n")

    # pass the arguments to the trainer
    train(training_data=training_data,
            validation_data=validation_data,
            batch_size=config.batch_size, 
            nb_epochs=config.num_epochs,
            min_lr=config.min_lr,
            max_lr=config.max_lr,
            save_dir_path=config.save_dir_path)
    

###########################################################################  


if __name__ == '__main__':
      main()  
