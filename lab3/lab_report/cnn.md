### **Lab Report: Convolutional Neural Network (CNN) for MNIST Image Classification**

#### **Objective:**
The objective of this lab is to implement and train a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The goal is to learn the process of building and training a CNN model using TensorFlow and evaluate its performance on a test set.

#### **Theory:**
A Convolutional Neural Network (CNN) is a deep learning algorithm commonly used for image processing tasks. It consists of layers that automatically learn features from raw pixel data. A typical CNN architecture includes convolutional layers, pooling layers, fully connected layers, and an output layer. CNNs are effective for image classification because they can capture spatial hierarchies of features.

The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits (0-9), each image being 28x28 pixels. The model is trained to classify these digits using the CNN architecture.

#### **Steps:**

1. **Import Libraries:**
   - Import necessary libraries including TensorFlow and Keras for building the neural network.
   
   ```python
   from tensorflow.keras.datasets import mnist
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
   ```

2. **Load and Preprocess Data:**
   - Load the MNIST dataset using Keras' `mnist.load_data()`.
   - Reshape the data to match the input format for CNNs, adding a channel dimension (1 for grayscale).
   - Normalize pixel values to range between 0 and 1 for better model convergence.
   
   ```python
   (X_train, y_train), (X_test, y_test) = mnist.load_data()
   X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)) / 255
   X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)) / 255
   ```

3. **Build the CNN Model:**
   - Create a Sequential model.
   - Add a convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation.
   - Add a max-pooling layer with a 2x2 pool size to reduce spatial dimensions.
   - Flatten the data and add a dense fully connected layer with 100 neurons and ReLU activation.
   - Add an output layer with 10 neurons (for 10 classes) and softmax activation for multi-class classification.
   
   ```python
   model = Sequential()
   model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
   model.add(MaxPool2D(2, 2))
   model.add(Flatten())
   model.add(Dense(100, activation='relu'))
   model.add(Dense(10, activation='softmax'))
   ```

4. **Compile the Model:**
   - Use the Adam optimizer and sparse categorical crossentropy loss function.
   - Track accuracy as the evaluation metric during training.
   
   ```python
   model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   ```

5. **Train the Model:**
   - Fit the model to the training data for 10 epochs.
   
   ```python
   model.fit(X_train, y_train, epochs=10)
   ```

6. **Evaluate the Model:**
   - After training, evaluate the model’s performance on the test data.
   
   ```python
   model.evaluate(X_test, y_test)
   ```

#### **Results:**
After training the model, the final evaluation will provide a loss value and accuracy. The accuracy indicates how well the model can classify handwritten digits from the MNIST dataset.

#### **Conclusion:**
In this lab, we successfully implemented a CNN for handwritten digit classification using the MNIST dataset. By following the steps of loading the dataset, building and training the CNN model, and evaluating its performance, we achieved a model capable of classifying digits with high accuracy.

---