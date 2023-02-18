import cv2
import numpy as np
import os
import tensorflow as tf

from sklearn.model_selection import train_test_split


EPOCHS = 10 # can edit min is 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 13
TEST_SIZE = 0.5 # can edit 0.0 - 0.9


def load_data():
    """
    Loads features and labels from the data folder.

    Returns:
        features: a numpy array of shape (num_samples, IMG_WIDTH * IMG_HEIGHT)
        labels: a numpy array of shape (num_samples,)
    """
    features = []
    labels = []
    for folder in os.listdir("data"):
        for image_path in os.listdir(os.path.join("data", folder)):
            # Read the image and resize it
            image = cv2.imread(os.path.join("data", folder, image_path), cv2.IMREAD_GRAYSCALE)
            img=~image
            if img is not None:
                ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
                ctrs,ret=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
                w=int(IMG_WIDTH)
                h=int(IMG_WIDTH)
                maxi=0
                for c in cnt:
                    x,y,w,h=cv2.boundingRect(c)
                    maxi=max(w*h,maxi)
                    if maxi==w*h:
                        x_max=x
                        y_max=y
                        w_max=w
                        h_max=h
                im_crop= thresh[y_max:y_max+h_max, x_max:x_max+w_max]
                im_resize = cv2.resize(im_crop,(IMG_WIDTH,IMG_WIDTH))

            # Flatten the image into a 1D array
            features.append(im_resize.flatten())

            # Add the label for this image
            if folder == '+':
                labels.append(int(10))
            elif folder == '-':
                labels.append(int(11))
            elif folder == '*':
                labels.append(int(12))
            else:
                labels.append(int(folder))

    return np.array(features), np.array(labels)


def build_model():
    """
    Builds and returns the model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(IMG_WIDTH * IMG_HEIGHT,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model


def main():
    # Load features and labels
    features, labels = load_data()

    # Split data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=TEST_SIZE
    )

    # Build the model
    model = build_model()

    # Train the model
    model.fit(
        train_features,
        train_labels,
        epochs=EPOCHS,
        batch_size=32,
        validation_data=(test_features, test_labels)
    )

    # Save the model
    model.save("handwritten_equation_solver.h5")


main()