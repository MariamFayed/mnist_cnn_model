import random
import os
import cv2 as cv
import numpy as np
from keras import Sequential
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Dataset Path
directory = "C:/Users/Mariam Fayed/PycharmProjects/project/TrainingSet"
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# Image Sizes are 28x28
image_size = 28
training_dataset = []

def Load_data(directory):
    # This loop passes on all subdirectories in the main directory
    for label in labels:
        path = os.path.join(directory, label)
        # Label class contains the indices of the subdirectories which corresponds to the labels.
        label_class = labels.index(label)
        for image in os.listdir(path):
            # Exception loop to handle any broken images.
            try:
                # Loading images to the training dataset array with its label attached to it.
                img_arr = cv.imread(os.path.join(path,image))
                training_dataset.append([img_arr, label_class])
            except Exception as e:
                pass
    # Shuffle all training data together.
    random.shuffle(training_dataset)
    training_data = []
    data_label = []

    # Divide the training set array into 2 arrays.
    # Training_data containing all the images
    # data_label containing the label for the image.
    for features, label in training_dataset:
        training_data.append(features)
        data_label.append(label)
    training_data = np.array(training_data).reshape(-1, image_size, image_size, 3)
    return training_data, data_label

def Train(images_array,label_array):

    # Splitting the data into Test and Train.
    X_train, X_test, y_train, y_test = train_test_split(images_array, label_array, test_size= 0.25, shuffle=True, random_state =30)

    # These two lines to check the number of Training Data and Testing Data after splitting.
    # And the size of the data set (28x28) with 3 inputs, as the images are used in an RGB Format.
    print("X_train shape : {}".format(X_train.shape))
    print("X_test shape  : {}".format(X_test.shape))

    # Normalization is a pre-processing step before structuring the model.
    # Convert from integers to floats
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # We need to normalize our data to be between 0 and 1 for prediction.
    X_train = np.divide(X_train, 255)
    X_test = np.divide(X_test, 255)

    # 10 Output Nodes for the 10 Possible Predicted Labels
    classes = 10
    # One hot encode Target Values where 1 for predicted class and 0 for the rest.
    y_train = to_categorical(y_train, classes)
    y_test = to_categorical(y_test, classes)

    # Now we create our CNN Model.
    # The CNN Model is based on a Frontend responsible for Training and extracting of freatures using the Convlutional & Pooling Layers.
    # Backend of the model is the Classifier of the model that makes the prediction.

    # Create an instance of the model.
    model = Sequential()
    # Create 1 convolutional layer with 32 filters which are the no. of filters the layer will learn with a kernel filter 3x3
    # Input Image size = 28x28 and a channel size 3
    # An Output image of 26x26 will be produced of this function.
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 3)))
    # Max Pooling is applied to produce with a 2x2 Pooling Window, stride of 2; taking the Max of each and producing a 13x13 image.
    model.add(MaxPooling2D((2, 2)))
    # Flatting the output into a 1D to enter the fully connected neural network.
    model.add(Flatten())
    # Dense function is used to reduce the input nodes into 100.
    # The Rectified Linear Activation Function is used which outputs the input directly if +ve and 0 if otherwise. (Adds a sort of non-linearity)
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    # Using softmax activation function and setting 10 output nodes.
    model.add(Dense(10, activation='softmax'))

    # Using Stochastic Gradient Descent with a learning rate = 0.01 and Momentum = 0.9.
    opt = SGD(lr=0.01, momentum=0.9)

    # Optimized Loss Function of the model.
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # If the model trained before, try loading the model.

    # Tensorboard Section
    from keras.callbacks import TensorBoard
    import time
    model_name = "MNIST-CNN-{}".format(int(time.time()))
    tb = TensorBoard(log_dir="logs\\{}".format(model_name))

    # Model Fitting using 20 iterations, and using the splitted test data as validation to prevent overfitting.
    # Callback for tensorboard variable and saving in the logs folder where tensorboard will read from.
    model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_test, y_test), callbacks=[tb])
    model.save("model.h5")
    model.summary()

    # Testing Predictions on the Trained model and Comparing them to their actual Label.
    prediction = model.predict_classes(X_test[:4])
    for i in range(4):
        print("Predicted Value:", prediction[i], ";", "Actual Label for the Image:", np.argmax(y_test[i]))
        plt.imshow(X_test[i])
        plt.show()
    return model
# Load data function call.
images_array, label_array = Load_data(directory)
# Train data function call.
Train(images_array,label_array)


