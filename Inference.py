from tensorflow import keras
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def Predict(image_path):
    # Load the trained model.
    model_loaded = keras.models.load_model("model.h5")
    # Read the passed Image Path
    image = cv.imread(image_path)
    plt.imshow(image)
    plt.show()
    # Predict function of the model takes a 4D Image so we expand the image to be (1,28,28,3)
    # Where 1 is the sample enumeration since we are predicting 1 image only.
    image_expanded = np.expand_dims(image, axis=0)
    prediction = model_loaded.predict_classes(image_expanded)
    print("The predicted label is =", prediction)
    return prediction
# Image [2]
Predict("img_100.jpg")