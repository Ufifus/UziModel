import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class Preprocessor:
    """Preprocessing images before upload in model"""

    def __init__(self, csv_file_, size_img_=(300, 300)):
        """1) image_size default=(300, 300)
        2) csv_file with dataset"""
        self.csv_file = csv_file_
        self.img_height = size_img_[0]
        self.img_wight = size_img_[1]

        print(self.img_height, self.img_wight)

    def upload_img(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

    def change_img(self, img):
        resize_and_rescale = tf.keras.Sequential([
            tf.keras.layers.Resizing(self.img_height, self.img_wight),
            tf.keras.layers.Rescaling(1. / 255)
        ])
        return resize_and_rescale(img, training=True)

    def augmentate_img(self, img):
       flip_and_rotation = tf.keras.Sequential([
           tf.keras.layers.RandomFlip("horizontal_and_vertical"),
           tf.keras.layers.RandomRotation(0.1),
        ])
       return flip_and_rotation(img, training=True)

    def create_dataset(self, n_classes, istraining=True):
        print(f'reading csv file with data {self.csv_file}...')
        old_dataset = pd.read_csv(self.csv_file)
        new_dataset = np.array([self.upload_img(img_path) for img_path in old_dataset['path']])
        print('Размер исходного набора -->', new_dataset.shape)
        new_dataset = np.array([self.change_img(img) for img in new_dataset])
        targets = [target for target in old_dataset['y']]

        if istraining:
            pass
        else:
            targets = self.create_logits(n_classes, targets)

        X_train, X_test, y_train, y_test = train_test_split(new_dataset,
                                                            np.array(targets),
                                                            test_size=0.2,
                                                            random_state=42)
        X_train = np.array([self.augmentate_img(img) for img in X_train])

        print('Размер тренировочного набора -->', X_train.shape)
        print('Размер тренировочный таргетов -->', y_train.shape)
        return X_train, X_test, y_train, y_test

    def create_logits(self, n_classes, y_targets):
        labels = []
        logits = [0] * n_classes
        for logit in y_targets:
            copy_logits = logits.copy()
            copy_logits[logit] = 1
            labels.append(copy_logits)
        return np.array(labels)

    def plot_imgs(self, imgs, titles=[]):
        ## single image
        if (len(imgs) == 1) or (type(imgs) not in [list, pd.core.series.Series]):
            img = imgs if type(imgs) is not list else imgs[0]
            title = None if len(titles) == 0 else (titles[0] if type(titles) is list else titles)
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.suptitle(title, fontsize=15)
            if len(img.shape) > 2:
                plt.imshow(img)
            else:
                plt.imshow(img, cmap=plt.cm.binary)

        ## multiple images
        else:
            fig, ax = plt.subplots(nrows=1, ncols=len(imgs), sharex=False, sharey=False, figsize=(4 * len(imgs), 10))
            if len(titles) == 1:
                fig.suptitle(titles[0], fontsize=15)
            for i, img in enumerate(imgs):
                ax[i].imshow(img)
                if len(titles) > 1:
                    ax[i].set(title=titles[i])
        plt.show()

    # def convert_to_tensor(self, dataset, targets, shuffle, augment, len_dataset, batch_size):
    #     AUTOTUNE = tf.data.AUTOTUNE
    #     tensor_dataset = tf.data.Dataset.from_tensor_slices((dataset, targets))
    #
    #     new_dataset = tensor_dataset.map(lambda x, y: (self.change_img(x), y), num_parallel_calls=AUTOTUNE)
    #     if shuffle:
    #         new_dataset = new_dataset.shuffle(len_dataset)
    #     new_dataset = new_dataset.batch(batch_size)
    #     if augment:
    #         new_dataset = new_dataset.map(lambda x, y: (self.augmentate_img(x), y), num_parallel_calls=AUTOTUNE)
    #     return new_dataset.prefetch(buffer_size=AUTOTUNE)