from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# TODO 1: Normalize the data to the range [0, 1]
X_train = X_train.astype('float32') / 255.0

# TODO 2: Normalize the testing data
X_test = X_test.astype('float32') / 255.0

# TODO 3: One-hot encode the training labels (hint to_categorical)
y_train = to_categorical(y_train, 10)

# TODO 4: One-hot encode the testing labels
y_test = to_categorical(y_test, 10)

# Build the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the input image to a 1D array
    Dense(128, activation='relu'),  # First hidden layer with 128 neurons and ReLU activation
    Dense(64, activation='relu'),  # Second hidden layer with 64 neurons and ReLU activation
    Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each class) and softmax activation
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=200, validation_split=0.2, verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Accuracy: {accuracy * 100:.2f}%')

# TODO 5: Make predictions (hint: model's predict function)
predictions = model.predict(X_test)

# Visualize the first 5 test samples and their predicted and actual labels
for i in range(5):
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f'Predicted: {predictions[i].argmax()}, Actual: {y_test[i].argmax()}')
    plt.show()
