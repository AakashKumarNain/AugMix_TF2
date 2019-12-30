import os
import time
import logging
import multiprocessing
import numpy as np

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import OrderedEnqueuer
from models.resnet20 import resnet_v1
from data_generator import DataGenerator
from utils import CTLEarlyStopping


###################################################################################

# get the model instance
print("\nLoading model")
model = resnet_v1(input_shape=(32, 32, 3), num_classes=10)
model.summary()
print("")

# loss function to be used for training
kld = tf.keras.losses.KLDivergence()
entropy = tf.keras.losses.CategoricalCrossentropy()

# metric to keep track of 
train_accuracy = tf.keras.metrics.CategoricalAccuracy()
test_accuracy = tf.keras.metrics.CategoricalAccuracy()
train_loss = tf.keras.metrics.Mean()
test_loss = tf.keras.metrics.Mean()

# initialize total steps to None and update it from main file
total_steps = None

# same for min_lr and max_lr
lr_max, lr_min = None, None

# earlystopping for custom training loops
es = CTLEarlyStopping(monitor="val_loss", mode="min", patience=5)

###################################################################################


def jsd_loss_fn(y_true, y_pred_clean, y_pred_aug1, y_pred_aug2):
    # cross entropy loss that is used for clean images only
    loss = entropy(y_true, y_pred_clean)

    mixture = (y_pred_clean + y_pred_aug1 + y_pred_aug2) / 3.
    mixture = tf.math.log(tf.clip_by_value(mixture, 1e-7, 1.))

    loss += 12. * (kld(mixture, y_pred_clean) + 
                    kld(mixture, y_pred_aug1) +
                    kld(mixture, y_pred_aug2)) / 3.
    return loss

###################################################################################

# learning rate scheduler
def get_lr(step):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))

###################################################################################

@tf.function
def train_step(clean, aug1, aug2, labels, optim):
    with tf.GradientTape() as tape:
        # get predictions on clean images
        y_pred_clean = model(clean, training=True)
        
        # get predictions on augmented images
        y_pred_aug1 = model(aug1, training=True)
        y_pred_aug2 = model(aug2, training=True)

        # calculate loss
        loss_value = jsd_loss_fn(y_true = labels, 
                            y_pred_clean = y_pred_clean,
                            y_pred_aug1 = y_pred_aug1,
                            y_pred_aug2 = y_pred_aug2)
        
    grads = tape.gradient(loss_value, model.trainable_weights)
    optim.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value, y_pred_clean


###################################################################################

@tf.function
def validate_step(images, labels):
    y_pred = model(images, training=False)
    loss = entropy(labels, y_pred)
    return loss, y_pred



###################################################################################

def train(training_data, 
            validation_data, 
            batch_size=32, 
            nb_epochs=100,
            min_lr=1e-5,
            max_lr=1.0,
            save_dri_path=""):
    

    x_train, y_train, y_train_cat = training_data
    x_test, y_test, y_test_cat = validation_data
    test_indices = np.arange(len(x_test))

    # get the training data generator. We are not using validation generator because the 
    # data is already loaded in memory and we don't have to perform any extra operation 
    # apart from loading the validation images and validation labels.
    ds = DataGenerator(x_train, y_train_cat, batch_size=batch_size)
    enqueuer = OrderedEnqueuer(ds, use_multiprocessing=True)
    enqueuer.start(workers=multiprocessing.cpu_count())
    train_ds = enqueuer.get()

    # get the total number of training and validation steps
    nb_train_steps = int(np.ceil(len(x_train) / batch_size))
    nb_test_steps = int(np.ceil(len(x_test) / batch_size))
    
    global total_steps, lr_max, lr_min
    total_steps = nb_train_steps
    lr_max = max_lr
    lr_min = min_lr
    
    # get the optimizer
    optim = optimizers.SGD(learning_rate=get_lr(0))
    
    print("Starting training: \n")
    for epoch in range(nb_epochs):
        start = time.time()
        # Train for an epoch and keep track of 
        # loss and accracy for each batch.
        for bno, (images, labels) in enumerate(train_ds):
            if bno==nb_train_steps:
                break
            # Get the batch data 
            clean, aug1, aug2 = images
            loss_value, y_pred_clean = train_step(clean, aug1, aug2, labels, optim)

            # Record batch loss and batch accuracy
            train_loss(loss_value)
            train_accuracy(labels, y_pred_clean)

    
        # Validate after each epoch
        for bno in range(nb_test_steps):
            # Get the indices for the current batch
            indices = test_indices[bno*batch_size:(bno + 1)*batch_size]
            
            # Get the data 
            images, labels = x_test[indices], y_test_cat[indices]

            # Get the predicitions and loss for this batch
            loss_value, y_pred = validate_step(images, labels)

            # Record batch loss and accuracy
            test_loss(loss_value)
            test_accuracy(labels, y_pred)

        
        # get training and validataion stats
        # after one epoch is completed 
        loss = train_loss.result()
        acc =  train_accuracy.result()
        val_loss = test_loss.result()
        val_acc = test_accuracy.result()


        # print loss values and accuracy values for each epoch 
        # for both training as well as validation sets
        print(f"""Epoch: {epoch+1} 
                train_loss: {loss:.6f}  train_acc: {acc*100:.2f}%  
                test_loss:  {val_loss:.6f}  test_acc:  {val_acc*100:.2f}%""")
        print("")

        # check progress using earlystopping
        if es.check_progress(val_loss):
            break
                
        
        # Reset the losses and accuracy
        train_loss.reset_states() 
        train_accuracy.reset_states()
        test_loss.reset_states() 
        test_accuracy.reset_states()
        
        end = time.time()
        print(f"Time taken for epoch {epoch+1}:  {end-start:.3f} seconds")
        print("="*78)