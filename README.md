# Digit_Recognition
This code trains a convolutional neural network (CNN) to classify hand written digits using the MNIST data set.  The resulting weights of the CNN are saved in h5 format ("CNN_weights_digit_recognition.h5").

The key steps in the model are as follows:
1. Training and test data sets are created from the MNIST dataset
2. The training data is augmented by randomly rotating, zooming in, blurring and translating the images
3. Examples training data images (following augmentation) are shown
4. The CNN architecture, the optimiser and the learning rate are defined.
5. The CNN is trained and the resulting weights are saved in h5 format.
6. The training history of the model is shown.
7. The accuracy of the model is calculated using the test set.

