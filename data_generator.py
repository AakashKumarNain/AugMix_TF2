import os
import logging
import numpy as np
from PIL import Image
from PIL import ImageOps

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.utils import Sequence, to_categorical
from augmentation import augmentations

##########################################################################

class DataGenerator(Sequence):
    def __init__(self, 
                 data, 
                 labels,
                 img_dim=(32, 32,3), 
                 batch_size=32, 
                 num_classes=10, 
                 shuffle=True,
                 jsd=True
                ):
        
        self.data = data
        self.labels = labels
        self.img_dim = img_dim
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.jsd = jsd
        self.augmentations = augmentations
        self.on_epoch_end()
        

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)


    def apply_op(self, image, op, severity):
        image = np.clip(image * 255., 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image)  # Convert to PIL.Image
        pil_img = op(pil_img, severity)
        return np.asarray(pil_img).astype(np.float32) / 255.


    def augment_and_mix(self, image, severity=3, width=3, depth=-1, alpha=1.):
        """Perform AugMix augmentations and compute mixture.
        Args:
            image: Raw input image as ndarray shape (h, w, c)
            severity: Severity of underlying augmentation operators (1-10).
            width: Width of augmentation chain
            depth: Depth of augmentation chain. -1 or (1, 3)
            alpha: Probability coefficient for Beta and Dirichlet distributions.
        Returns:
            mixed: Augmented and mixed image.
        """
        ws = np.random.dirichlet([alpha] * width).astype(np.float32)
        m  = np.float32(np.random.beta(alpha, alpha))
        mix = np.zeros_like(image).astype(np.float32)

        for i in range(width):
            image_aug = image.copy()
            depth = depth if depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.augmentations)
                image_aug = self.apply_op(image_aug, op, severity)
                # Preprocessing commutes since all coefficients are convex
                mix += ws[i] * image_aug

        # mix the image and return 
        mixed = (1 - m)*image + m*mix
        return mixed


    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))
    
    
    def __getitem__(self, idx):
        curr_batch = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_len = len(curr_batch)  
        X_orig = np.zeros((batch_len, *self.img_dim), dtype=np.float32)
        y = np.zeros((batch_len, self.num_classes), dtype=np.float32)
        
        if self.jsd:
            X_aug1 = np.zeros_like(X_orig, dtype=np.float32)
            X_aug2 = np.zeros_like(X_orig, dtype=np.float32)

        for i, index in enumerate(curr_batch):
            img = self.data[index]
            X_orig[i] = img
            if self.jsd:
                X_aug1[i] = self.augment_and_mix(img)
                X_aug2[i] = self.augment_and_mix(img)
            y[i] = self.labels[index]
        
        if not self.jsd:
            return X_orig, y
        else:
            return [X_orig, X_aug1, X_aug2], y