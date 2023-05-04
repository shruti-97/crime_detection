import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score

from IPython.display import clear_output
import warnings

warnings.filterwarnings('ignore')


class CrimeDetection():
    def __init__(self):
        # Define all the variables
        self.test_directory = None
        self.train_directory = None
        self.y_test = None
        self.predictions = None
        self.model = None
        self.test_generator = None
        self.train_generator = None
        self.test_metadata = None
        self.train_metadata = None
        self.preprocessing_function = None
        self.CLASS_LABELS = None
        self.NUM_CLASSES = None
        self.LR = None
        self.EPOCHS = None
        self.BATCH_SIZE = None
        self.IMG_WIDTH = None
        self.IMG_HEIGHT = None
        self.SEED = None


    def meta_data(self):
        self.train_directory = "./archive/Train"
        self.test_directory = "./archive/Test"

        self.SEED = 12
        self.IMG_HEIGHT = 64
        self.IMG_WIDTH = 64
        self.BATCH_SIZE = 64
        self.EPOCHS = 1
        self.LR = 0.00003
        self.NUM_CLASSES = 14
        self.CLASS_LABELS = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', "Normal",
                             'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']

    def generate_data(self):
        self.preprocessing_function = tf.keras.applications.densenet.preprocess_input

        self.train_metadata = ImageDataGenerator(horizontal_flip=True,
                                                 width_shift_range=0.1,
                                                 height_shift_range=0.05,
                                                 rescale=1. / 255,
                                                 preprocessing_function=self.preprocessing_function
                                                 )
        self.test_metadata = ImageDataGenerator(rescale=1. / 255,
                                                preprocessing_function=self.preprocessing_function
                                                )

    def get_data(self):
        self.train_generator = self.train_metadata.flow_from_directory(directory=self.train_directory,
                                                                       target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                                       batch_size=self.BATCH_SIZE,
                                                                       shuffle=True,
                                                                       color_mode="rgb",
                                                                       class_mode="categorical",
                                                                       seed=self.SEED
                                                                       )
        self.test_generator = self.test_metadata.flow_from_directory(directory=self.test_directory,
                                                                     target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                                     batch_size=self.BATCH_SIZE,
                                                                     shuffle=False,
                                                                     color_mode="rgb",
                                                                     class_mode="categorical",
                                                                     seed=self.SEED
                                                                     )

    def plot_data(self, data):
        fig = px.bar(x=self.CLASS_LABELS,
                     y=[list(data.classes).count(i) for i in np.unique(data.classes)],
                     color=np.unique(data.classes),
                     color_continuous_scale="Emrld")
        fig.update_xaxes(title="Classes")
        fig.update_yaxes(title="Number of Images")
        fig.update_layout(showlegend=True,
                          title={
                              'text': 'Test Data Distribution ',
                              'y': 0.95,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'})
        fig.show()

    def feature_extractor(self, inputs):
        feature_extractor = tf.keras.applications.DenseNet121(input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3),
                                                              include_top=False,
                                                              weights=None)(inputs)

        return feature_extractor

    def classifier(self, inputs):
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(self.NUM_CLASSES, activation="softmax", name="classification")(x)

        return x

    def final_model(self, inputs):
        densenet_feature_extractor = self.feature_extractor(inputs)
        classification_output = self.classifier(densenet_feature_extractor)

        return classification_output

    def train_model(self, model_name):
        self.model.fit(x=self.train_generator, validation_data=self.test_generator, epochs=self.EPOCHS)
        self.model.save(model_name)

    def define_compile_model(self):
        inputs = tf.keras.layers.Input(shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3))
        classification_output = self.final_model(inputs)
        self.model = tf.keras.Model(inputs=inputs, outputs=classification_output)

        self.model.compile(optimizer=tf.keras.optimizers.SGD(self.LR),
                           loss='categorical_crossentropy',
                           metrics=[tf.keras.metrics.AUC()])

    def test_model(self):
        self.predictions = self.model.predict(self.test_generator)
        self.y_test = self.test_generator.classes

    def multiclass_roc_auc_score(self, average="macro"):
        fig, c_ax = plt.subplots(1, 1, figsize=(15, 8))
        lb = LabelBinarizer()
        lb.fit(self.y_test)
        y_test = lb.transform(self.y_test)
        for (idx, c_label) in enumerate(self.CLASS_LABELS):
            fpr, tpr, thresholds = roc_curve(y_test[:, idx].astype(int), self.predictions[:, idx])
            c_ax.plot(fpr, tpr, lw=2, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
        c_ax.plot(fpr, fpr, 'black', linestyle='dashed', lw=4, label='Random Guessing')
        plt.xlabel('FALSE POSITIVE RATE', fontsize=18)
        plt.ylabel('TRUE POSITIVE RATE', fontsize=16)
        plt.legend(fontsize=11.5)
        plt.show()
        return roc_auc_score(y_test, self.predictions, average=average)

    def accuracy(self):
        y_preds = np.argmax(self.predictions, axis=1)
        return accuracy_score(self.y_test.astype(int), y_preds)



if __name__ == "__main__":
    crime_detection = CrimeDetection()
    crime_detection.meta_data()
    crime_detection.generate_data()
    crime_detection.get_data()

    # Plot for train
    crime_detection.plot_data(crime_detection.train_generator)

    # Plot for test
    crime_detection.plot_data(crime_detection.test_generator)

    crime_detection.define_compile_model()
    clear_output()
    print(crime_detection.model.summary())

    # Train the model
    crime_detection.train_model('ucf_crime')

    # Test the model
    crime_detection.test_model()

    print('ROC AUC score:', crime_detection.multiclass_roc_auc_score(average="micro"))
    print('Accuracy score:', crime_detection.accuracy())
