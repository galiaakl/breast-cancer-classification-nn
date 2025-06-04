import numpy as np
from numpy import random
import os
import pickle
 
np.random.seed(42)

class NeuralNetwork:
    def __init__(self, layers, learning_rate, momentum_factor, epochs):
        self.layers = layers
        self.num_layers = len(layers)
        self.learning_rate = learning_rate
        self.momentum_factor = momentum_factor
        self.epochs = epochs
        self.weights = [np.random.randn(layers[i-1], layers[i]) * 0.1 for i in range(1, self.num_layers)]  # Smaller initial weights
        self.old_weights = [np.zeros((layers[i-1], layers[i])) for i in range(1, self.num_layers)]

    def save_weights(self, filename="weights.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.weights, f)

    def load_weights(self, filename="weights.pkl"):
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                self.weights = pickle.load(f)
        else:
            print("File not found. Loading default weights.")

    def sigmoid(self, z):
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward_propagation(self, x):
        activations = [x]  # assign input to be activated
        zs = []  # effective Y in the next layer

        for w in self.weights:
            z = np.dot(activations[-1], w)
            zs.append(z)
            activations.append(self.sigmoid(z))

        return activations, zs  # V_layer, Y_layer

    def backward_propagation(self, x, y, activations, zs):
        deltas = [None] * self.num_layers
        gradients_w = [np.zeros(w.shape) for w in self.weights]

        # Calculate output layer error
        deltas[-1] = (y - activations[-1]) * self.sigmoid_derivative(zs[-1])  # output_gradient

        # Backpropagation for hidden layers
        for l in range(self.num_layers - 2, 0, -1):  # Exclude input and output layers
            deltas[l] = np.dot(deltas[l+1], self.weights[l].T) * self.sigmoid_derivative(zs[l-1])

        # Compute gradients
        for l in range(self.num_layers - 1):
            gradients_w[l] = np.dot(activations[l].reshape(-1, 1), deltas[l+1].reshape(1, -1))

        return gradients_w

    def train(self, X, y):
        highest_accuracy = 0  # Variable to track the highest accuracy
        for epoch in range(self.epochs):
            total_gradients_w = [np.zeros(w.shape) for w in self.weights]
            correct_predictions = 0

            for x, y_target in zip(X, y):
                x = np.array(x).reshape(-1, 1).T
                y_target = np.array(y_target).reshape(-1, 1)

                activations, zs = self.forward_propagation(x)
                gradients_w = self.backward_propagation(x, y_target, activations, zs)

                # Accumulate gradients
                total_gradients_w = [tw + gw for tw, gw in zip(total_gradients_w, gradients_w)]

                # Check if our prediction is correct
                prediction = 1 if activations[-1][0] >= 0.5 else 0
                if prediction == y_target:
                    correct_predictions += 1

            # Calculate accuracy
            accuracy = (correct_predictions / len(X))
            print(f"Epoch {epoch+1}/{self.epochs}, Accuracy: {accuracy*100:.2f}%", flush=True)

            # Update highest_accuracy if current accuracy is higher
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy

            # FIXED: Update weights with gradients AND momentum
            for l in range(self.num_layers - 1):
                # Calculate momentum term
                momentum = self.momentum_factor * (self.weights[l] - self.old_weights[l])
                
                # Store current weights before updating
                self.old_weights[l] = self.weights[l].copy()
                
                # Update weights: learning_rate * average_gradient + momentum
                self.weights[l] += self.learning_rate * (total_gradients_w[l] / len(X)) + momentum

        print(f"Training completed. Highest accuracy achieved: {highest_accuracy*100:.2f}%")
        
        
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Implementation script
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data - change this to your actual file name
data = pd.read_csv("data_BreastCancer.csv")
X = data.iloc[:, 2:].values    # Features
y = data.iloc[:, 1].values     # Target

X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))  # normalize inputs

y = [1 if val == 'M' else 0 for val in y]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))

# Define network architecture
layers = [30, 16, 1]  
learning_rate = 0.05
momentum_factor = 0.95
epochs = 2000          

# Create neural network
nn = NeuralNetwork(layers, learning_rate, momentum_factor, epochs)

# Train the network
nn.train(X_train, y_train)
print("\n" + "="*50)
print("ACCURACY RESULTS")
print("="*50)

# Calculate training accuracy
train_predictions = []
for x in X_train:
    x = np.array(x).reshape(-1, 1).T
    activations, _ = nn.forward_propagation(x)
    prediction = 1 if activations[-1][0] >= 0.5 else 0
    train_predictions.append(prediction)

train_accuracy = sum(p == t for p, t in zip(train_predictions, y_train)) / len(y_train)

# Calculate test accuracy  
test_predictions = []
for x in X_test:
    x = np.array(x).reshape(-1, 1).T
    activations, _ = nn.forward_propagation(x)
    prediction = 1 if activations[-1][0] >= 0.5 else 0
    test_predictions.append(prediction)

test_accuracy = sum(p == t for p, t in zip(test_predictions, y_test)) / len(y_test)

# Display results
print(f"Training Accuracy: {train_accuracy*100:.2f}%")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Overfitting Score: {(train_accuracy - test_accuracy)*100:.2f}%")
print("="*50)


# Calculate confusion matrix for test set
true_positive = sum((p == 1) and (t == 1) for p, t in zip(train_predictions, y_train))
false_positive = sum((p == 1) and (t == 0) for p, t in zip(train_predictions, y_train))
false_negative = sum((p == 0) and (t == 1) for p, t in zip(train_predictions, y_train))
true_negative = sum((p == 0) and (t == 0) for p, t in zip(train_predictions, y_train))

# Display confusion matrix
confusion_matrix = np.array([['', 'PP', 'PN'],
                            ['AP', true_positive, false_negative],
                            ['AN', false_positive, true_negative]])

print("Confusion Matrix for the training set:")
for row in confusion_matrix:
    print("\t".join(map(str, row)))


# Calculate confusion matrix for test set
true_positive = sum((p == 1) and (t == 1) for p, t in zip(test_predictions, y_test))
false_positive = sum((p == 1) and (t == 0) for p, t in zip(test_predictions, y_test))
false_negative = sum((p == 0) and (t == 1) for p, t in zip(test_predictions, y_test))
true_negative = sum((p == 0) and (t == 0) for p, t in zip(test_predictions, y_test))

# Display confusion matrix for test set
confusion_matrix = np.array([['', 'PP', 'PN'],
                            ['AP', true_positive, false_negative],
                            ['AN', false_positive, true_negative]])

print("Confusion Matrix for the test set:")
for row in confusion_matrix:
    print("\t".join(map(str, row)))
