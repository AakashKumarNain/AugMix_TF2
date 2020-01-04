import os
import logging
import numpy as np
import matplotlib.pyplot as plt

class CTLEarlyStopping:
    def __init__(self,
               monitor='val_loss',
               min_delta=0,
               patience=0,
               mode='auto',
               ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stop_training = False
        self.improvement = False

        if mode not in ['auto', 'min', 'max']:
            logging.warning('EarlyStopping mode %s is unknown, '
                      'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less
        
        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1
            
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
    
    
    def check_progress(self, current):
        if self.monitor_op(current - self.min_delta, self.best):
            print(f"{self.monitor} improved from {self.best:.4f} to {current:.4f}.", end=" ")
            self.best = current
            self.wait = 0
            self.improvement = True
        else:
            self.wait += 1
            self.improvement = False
            print(f"{self.monitor} didn't improve")
            if self.wait >= self.patience:
                print("Early stopping")
                self.stop_training = True
                
        return self.improvement, self.stop_training
    
    
##########################################################################################

    
class CTLHistory:
    def __init__(self,
                 filename="history.png",
                 save_dir='plots'):
        
        self.history = {'train_loss':[], 
                        "train_acc":[], 
                        "val_loss":[], 
                        "val_acc":[]}
        
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.plot_name = os.path.join(self.save_dir, filename)
    
   
  
    def update(self, train_stats, val_stats):
        train_loss, train_acc = train_stats
        val_loss, val_acc = val_stats
        
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(np.round(train_acc*100))
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(np.round(val_acc*100))
        
        
    def plot_and_save(self, initial_epoch=0):
        train_loss = self.history['train_loss']
        train_acc = self.history['train_acc']
        val_loss = self.history['val_loss']
        val_acc = self.history['val_acc']
        
        epochs = [(i+initial_epoch) for i in range(len(train_loss))]
        
        f, ax = plt.subplots(1, 2, figsize=(15,8))
        ax[0].plot(epochs, train_loss)
        ax[0].plot(epochs, val_loss)
        ax[0].set_title('loss progression')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('loss values')
        ax[0].legend(['training', 'validation'])
        
        ax[1].plot(epochs, train_acc)
        ax[1].plot(epochs, val_acc)
        ax[1].set_title('accuracy progression')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend(['training', 'validation'])
        
        plt.savefig(self.plot_name)
        plt.close()
