from __init__ import *
import os


dataset_dir = os.path.abspath('Datasets')
csv_file = os.path.abspath('csv_dataset.csv')

X_train = X_test = y_train = y_test = n_classes = None

def main():
    n_classes = upload_data(dataset_dir, csv_file)
    print('Started preprocessing data...')
    preprocessor = Preprocessor(csv_file, (300, 300))
    X_train, X_test, y_train, y_test = preprocessor.create_dataset(n_classes)
    return X_train, X_test, y_train, y_test, n_classes

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, n_classes = main()