import os
import logging
import numpy as np
from PIL import Image
from PIL import ImageOps

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.utils import Sequence, to_categorical

##########################################################################

class DataGenerator(Sequence):
    def __init__(self, 
                 data, 
                 labels,
                 img_dim=(32, 32,3), 
                 batch_size=32, 
                 num_classes=10, 
                 shuffle=True,
                 augment=False,
                 jsd=True):
        self.data = data
        self.labels = labels
        self.img_dim = img_dim
        self.IMAGE_SIZE = img_dim[0]
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment
        self.jsd = jsd
        self.indices = np.arange(len(data))
        self.on_epoch_end()
        self.augmentations = [self.autocontrast, 
                              self.equalize, 
                              self.posterize, 
                              self.rotate, 
                              self.solarize, 
                              self.shear_x, 
                              self.shear_y,
                              self.translate_x, 
                              self.translate_y]
        self.counter = 1
        self.total_steps = int(np.ceil(len(self.data) / self.batch_size))


    def int_parameter(self, level, maxval):
        """Helper function to scale `val` between 0 and maxval .
        Args:
            level: Level of the operation that will be in [0, `PARAMETER_MAX`].
            maxval: Maximum value that the operation can have. This will be 
                    scaled to level/PARAMETER_MAX.
        Returns:
            An int that results from scaling `maxval` according to `level`.
        """
        return int(level * maxval / 10)


    def float_parameter(self, level, maxval):
        """Helper function to scale `val` between 0 and maxval.
        Args:
            level: Level of the operation that will be in [0, `PARAMETER_MAX`].
            maxval: Maximum value that the operation can have. This will be 
                    scaled to level/PARAMETER_MAX.
        Returns:
            A float that results from scaling `maxval` according to `level`.
        """
        return float(level) * maxval / 10.


    def sample_level(self, n):
        return np.random.uniform(low=0.1, high=n)


    def autocontrast(self, pil_img, _):
        return ImageOps.autocontrast(pil_img)


    def equalize(self, pil_img, _):
        return ImageOps.equalize(pil_img)


    def posterize(self, pil_img, level):
        level = self.int_parameter(self.sample_level(level), 4)
        return ImageOps.posterize(pil_img, 4 - level)


    def rotate(self, pil_img, level):
        degrees = self.int_parameter(self.sample_level(level), 30)
        if np.random.uniform() > 0.5:
            degrees = -degrees
        return pil_img.rotate(degrees, resample=Image.BILINEAR)


    def solarize(self, pil_img, level):
        level = self.int_parameter(self.sample_level(level), 256)
        return ImageOps.solarize(pil_img, 256 - level)


    def shear_x(self, pil_img, level):
        level = self.float_parameter(self.sample_level(level), 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                Image.AFFINE, (1, level, 0, 0, 1, 0),
                                resample=Image.BILINEAR)


    def shear_y(self, pil_img, level):
        level = self.float_parameter(self.sample_level(level), 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                Image.AFFINE, (1, 0, 0, level, 1, 0),
                                resample=Image.BILINEAR)


    def translate_x(self, pil_img, level):
        level = self.int_parameter(self.sample_level(level), self.IMAGE_SIZE / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                Image.AFFINE, (1, 0, level, 0, 1, 0),
                                resample=Image.BILINEAR)


    def translate_y(self, pil_img, level):
        level = self.int_parameter(self.sample_level(level), self.IMAGE_SIZE / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                Image.AFFINE, (1, 0, 0, 0, 1, level),
                                resample=Image.BILINEAR)


    def on_epoch_end(self):
        self.counter = 1 
        if self.shuffle:
            np.random.shuffle(self.indices)


    def apply_op(self, image, op, severity):
        image = np.clip(image * 255., 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(image)  # Convert to PIL.Image
        pil_img = op(pil_img, severity)
        return np.asarray(pil_img, dtype=np.float32) / 255.


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
        m  = np.random.beta(alpha, alpha)
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
        return self.total_steps
    
    def __getitem__(self, idx):
        curr_batch = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_len = len(curr_batch)
        
        if not self.jsd:
            X_orig = np.zeros((batch_len, *self.img_dim), dtype=np.float32)
            y = np.zeros((batch_len, 10), dtype=np.float32)
        else:
            X_orig = np.zeros((batch_len, *self.img_dim), dtype=np.float32)
            X_aug1 = np.zeros((batch_len, *self.img_dim), dtype=np.float32)
            X_aug2 = np.zeros((batch_len, *self.img_dim), dtype=np.float32)
            y = np.zeros((batch_len, 10), dtype=np.float32)

        for i, index in enumerate(curr_batch):
            img = self.data[index]
            X_orig[i] = self.augment_and_mix(img)
            if self.jsd:
                X_aug1[i] = self.augment_and_mix(img)
                X_aug2[i] = self.augment_and_mix(img)
            y[i] = self.labels[index]
        
        self.counter +=1
        if self.counter >=self.total_steps:
            self.on_epoch_end()
        
        if not self.jsd:
            return X_orig, y
        else:
            return [X_orig, X_aug1, X_aug2], y
