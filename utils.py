import logging
import numpy as np

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
            self.best = current
            print(f"{self.monitor} improved")
            self.wait = 0
        else:
            self.wait += 1
            print(f"{self.monitor} didn't improve")
            if self.wait >= self.patience:
                print("Early stopping")
                self.stop_training = True
                
        return self.stop_training