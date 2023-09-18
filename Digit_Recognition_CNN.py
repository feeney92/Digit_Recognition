import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from scipy.ndimage.filters import gaussian_filter

# Load the MNIST data set
(x_train, labels_train), (x_test, labels_test) = mnist.load_data()

# Convert the data set to the correct data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Scale the data set so that inputs are between 0 and 1
x_train /= 255
x_test /= 255

# Create the categorical target labels
y_train = to_categorical(labels_train, 10)
y_test = to_categorical(labels_test, 10)

# Convert the data to the right shape so it can be used as input into the CNN
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# Define a function which adds blur to the images (for use in the data augmentation function below)
def blur(img):
    rdm = np.random.uniform(0, 1)
    if rdm < 0.1:
        return gaussian_filter(img, sigma=1)
    elif rdm < 0.125:
        return gaussian_filter(img, sigma=1.5)
    else:
        return img


# Create augmented data by rotating, zooming in, blurring and translating the images
augmented_data = image.ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=20,  # randomly rotate images in the range
        zoom_range=0.1,  # randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,
        vertical_flip=False,
        preprocessing_function=blur)  # randomly blur image

# Add the augmented data to the training data set
augmented_data.fit(x_train)
it = augmented_data.flow(x_train, y_train, shuffle=False)
batch_images, batch_labels = next(it)


# Function for visualising the input data
def visualize_data(images, categories, class_names):
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor('white')
    for i in range(3 * 7):
        plt.subplot(3, 7, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i])
        class_index = categories[i].argmax()
        plt.xlabel(class_names[class_index])
    plt.show()


class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
visualize_data(batch_images, batch_labels, class_names)


# Create the convolution neural network ('CNN')
net = Sequential()
# Add filter layer with regularisation term
net.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=l2(0.0005)))
# Add filter layer
net.add(Conv2D(filters=32, kernel_size=(5, 5), use_bias=False))
# Apply batch normalisation
net.add(BatchNormalization())
# Add max pooling layer
net.add(MaxPool2D(pool_size=(2, 2)))
# Add dropout layer
net.add(Dropout(rate=0.25))
# Add filter layer with regularisation term
net.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.0005)))
# Add filter layer
net.add(Conv2D(64, (3, 3), activation='relu', use_bias=False))
# Apply batch normalisation
net.add(BatchNormalization())
# Add max pooling layer
net.add(MaxPool2D(pool_size=(2, 2)))
# Add dropout layer
net.add(Dropout(rate=0.25))
# Flatten output for use in dense layer
net.add(Flatten())
# Add dense layer
net.add(Dense(256, activation='relu'))
# Apply batch normalisation
net.add(BatchNormalization())
# Add dense layer
net.add(Dense(128, activation='relu'))
# Apply batch normalisation
net.add(BatchNormalization())
# Add dense layer
net.add(Dense(84, activation='relu'))
# Apply batch normalisation
net.add(BatchNormalization())
# Add dropout layer
net.add(Dropout(rate=0.25))
# Add softmax layer
net.add(Dense(10, activation='softmax'))

# Define the optimizer for the CNN
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Define the conditions under which the learning rate size is reduced
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

# Train the CNN (for 45 epochs)
net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
history = net.fit(it,
                  validation_data=(x_test, y_test),
                  epochs=45,
                  batch_size=256,
                  verbose=2,
                  callbacks=[learning_rate_reduction])

# Save the CNN weights once training has finished
net.save("CNN_weights_digit_recognition.h5")

# View the training history
plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Test the network
outputs = net.predict(x_test)
labels_predicted = np.argmax(outputs, axis=1)
misclassified = sum(labels_predicted != labels_test)
print('Percentage misclassified = ', 100*misclassified/labels_test.size)
