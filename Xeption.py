from keras.applications import xception
from tensorflow.keras import models, layers, utils, callbacks #(2.6.0)
from tensorflow import keras
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd



class XeptionModel:
    def __init__(self, model_=None):
        self.n_classes = None
        self.model = model_

    @classmethod
    def create_model(cls, n_classes):
        xept = xception.Xception(
                             weights='imagenet',
                             include_top=False,
                             input_shape=(300, 300, 3)
        )
        xept.trainable = False

        model = models.Sequential(name="XeptionLearning", layers=[
            xept,
            layers.Flatten(name="flat"),
            layers.Dense(name="dense", units=128, activation='relu'),
            layers.Dense(name="y_out", units=n_classes, activation="softmax")
        ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return cls(model)

    @classmethod
    def load_model(cls, dir):
        model = models.load_model(dir)
        return cls(model)

    def print_strucher(self):
        return print(self.model.summary())

    def training_model(self, X_train, y_train, epochs=10, val=0.2, verbose=1, batch_size=32, shuffle=True):
        training = self.model.fit(x=X_train, y=y_train, batch_size=batch_size, shuffle=shuffle,
                             epochs=epochs, verbose=verbose, validation_split=val,
                             callbacks=[callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)])
        self.model = training.model
        self.utils_plot_training(training)

    def funetine_model(self, X_train, y_train, epochs=10, val=0.2, verbose=1, batch_size=32, shuffle=True):
        self.model.trainable = True
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy()],
        )
        training = self.model.fit(x=X_train, y=y_train, batch_size=batch_size, shuffle=shuffle,
                                epochs=epochs, verbose=verbose, validation_split=val)
        self.model = training.model
        self.utils_plot_training(training)

    def test_model(self, X_test, y_test):
        predicted_prob = self.model.predict(X_test)
        predicted = [np.argmax(pred) for pred in predicted_prob]
        return self.evaluate_multi_classif(y_test, predicted, predicted_prob, figsize=(15, 5))

    def utils_plot_training(self, results):
        metrics = [k for k in results.history.keys() if ("loss" not in k) and ("val" not in k)]
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 3))

        ## training
        ax[0].set(title="Обучение")
        ax11 = ax[0].twinx()
        ax[0].plot(results.history['loss'], "o-", color='black')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss', color='black')
        for metric in metrics:
            ax11.plot(results.history[metric], "o-", label=metric)
        ax11.set_ylabel("Score", color='steelblue')
        ax11.legend()

        ## validation
        ax[1].set(title="Валидация")
        ax22 = ax[1].twinx()
        ax[1].plot(results.history['val_loss'], "o-", color='black')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss', color='black')
        for metric in metrics:
            ax22.plot(results.history['val_' + metric], "o-", label=metric)
        ax22.set_ylabel("Score", color="steelblue")
        plt.show()

    def evaluate_multi_classif(self, y_test, predicted, predicted_prob, figsize=(15, 5)):
        classes = np.unique(y_test)
        y_test_array = pd.get_dummies(y_test, drop_first=False).values

        ## Accuracy, Precision, Recall
        accuracy = metrics.accuracy_score(y_test, predicted)
        auc = metrics.roc_auc_score(y_test, predicted_prob, multi_class="ovr")
        print("Thyroid ultrasound image classification by Tirads Level")
        print("Accuracy:", round(accuracy, 2))
        print("Auc:", round(auc, 2))
        print("Detail:")
        print(metrics.classification_report(y_test, predicted))

        ## Plot confusion matrix
        cm = metrics.confusion_matrix(y_test, predicted)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Purples, cbar=False)
        ax.set(xlabel="Predict", ylabel="Fact", xticklabels=classes, yticklabels=classes,
               title="Tirads Predicitions Confusion matrix")
        plt.yticks(rotation=0)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        ## Plot roc
        for i in range(len(classes)):
            fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:, i], predicted_prob[:, i])
            ax[0].plot(fpr, tpr, lw=3, label='{0} (S={1:0.2f})'.format(classes[i], metrics.auc(fpr, tpr)))
        ax[0].plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
        ax[0].set(xlim=[-0.05, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate',
                  ylabel="True Positive Rate (Recall)", title="Receiver operating characteristic")
        ax[0].legend(loc="lower right")
        ax[0].grid(True)

        ## Plot precision-recall curve
        for i in range(len(classes)):
            precision, recall, thresholds = metrics.precision_recall_curve(y_test_array[:, i], predicted_prob[:, i])
            ax[1].plot(recall, precision, lw=3, label='{0} (area={1:0.2f})'.format(
                classes[i], metrics.auc(recall, precision)))
        ax[1].set(xlim=[0.0, 1.05], ylim=[0.0, 1.05], xlabel='Recall', ylabel="Precision",
                  title="Precision-Recall curve")
        ax[1].legend(loc="best")
        ax[1].grid(True)
        plt.show()

    def save_model(self, model, dir):
        model.save(dir)
        return f'Model successfuly saved in {dir}'