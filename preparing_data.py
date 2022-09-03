import pandas as pd
import os
import cv2
import numpy as np



able_formats = ['png', 'jpg', 'jpeg', 'JPG', 'bmp']
able_dirs = ['Tiroides1', 'Tiroides2', 'Tiroides3', 'Tiroides4', 'Tiroides5']


def upload_data(dirs_path, csv_file):
    """Upload data from our local computer:
    Get path to directory with datasets
    Return pandas datatable"""
    dataset = pd.DataFrame(columns=['name', 'path', 'level', 'y'])
    if not os.path.exists(dirs_path):
        return f'Directory "{dirs_path}" doesnt exist'
    else:
        for i, dir_name in enumerate(able_dirs):
            dir_path = os.path.join(dirs_path, dir_name)
            if os.path.exists(dir_path):
                images_names = [image_name for image_name in os.listdir(dir_path) for format in able_formats \
                                if image_name.endswith(format)]
                print(f'count of images in {dir_path} == {len(images_names)}')
                images_paths = [os.path.join(dir_path, image_name) for image_name in images_names]
                images_names = pd.DataFrame(images_names, columns=['name'])
                images_names['path'] = images_paths
                images_names['level'] = i + 1
                images_names['y'] = i
                print(images_names.head())
                dataset = dataset.append(images_names)

    print(dataset.head())
    n_classes = dataset["y"].nunique() #count of unique classes
    dataset.to_csv(csv_file)
    return n_classes


def upload_img(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(img)