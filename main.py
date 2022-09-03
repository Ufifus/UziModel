from __init__ import *
import os
from dataset import X_train, X_test, y_train, y_test, n_classes


model_dir = r"C:\Users\artem\Models\New_model_Xeption_2"
newmodel_dir = r"C:\Users\artem\Models\XeptionModel2"


def train_new_model():
    print('Started bring model...')
    model = XeptionModel.create_model(n_classes)
    model.print_strucher()
    print('Started fit model..')
    model.training_model(X_train, y_train)
    print('testing model...')
    model.test_model(X_test, y_test)
    return model


def train_exist_model():
    print('Started bring model...')
    model = XeptionModel.load_model(model_dir)
    model.print_strucher()
    print('Started fit model..')
    model.funetine_model(X_train, y_train)
    print('testing model...')
    model.test_model(X_test, y_test)
    return model


def main():
    if (X_train is None) or (y_train is None):
        return 'Doesnt exists training dataset'
    else:
        model = train_exist_model()
        issave = input('Input 1 if save model and 0 if not')
        if bool(issave):
            model.save_model(newmodel_dir)
        return None

if __name__ == '__main__':
    main()

