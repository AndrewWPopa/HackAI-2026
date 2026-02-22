import numpy as np
from datasets import load_dataset
import time
import json

ds = load_dataset("ylecun/mnist")

# Note: increasing layer number will cause accuracy to drop to 9.74% (Random)
input_size      = 784
output_size     = 10
epochs          = 10
batch_size      = 64
learning_rate   = 0.1
layer_array     = [7,7]

# Declare weights and biases list
weights = []
biases = []

# Concat input size and output size into layer_array
layer_size = [input_size] + layer_array + [output_size]

# Hidden layer next hidden layer (n_hidden amount of times)
for _ in range(len(layer_size) - 1):
    weights.append(np.random.randn(layer_size[_], layer_size[_+1]) * 0.1)
    biases.append(np.zeros(layer_size[_+1]))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

# Total time metric
totalStartTime = time.time()

# For all epochs
for epoch in range(epochs):
    epochStartTime = time.time()
    print(f"Epoch {epoch +1}")

    # Array's for images and labels
    batch_images = []
    batch_labels = []
    index = 0

    # Shuffled training data, with random seed as epoch counter
    shuffled_train = ds["train"].shuffle(seed=epoch)    
    for example in shuffled_train:

        # Flatten image into array and retrieve output value from example
        img = np.array(example["image"]).flatten() / 255.0
        label = example["label"]
        
        # Add the image and label of image to batch
        batch_images.append(img)
        batch_labels.append(label)
        index += 1
        
        # If batch has reached the size of the defined batch size, or the batch has reached the training data final index
        if len(batch_images) == batch_size or index == len(ds["train"]):
        
            # Convert to matrices
            X = np.stack(batch_images)
            y = np.zeros((len(batch_images), 10))
            y[np.arange(len(batch_images)), batch_labels] = 1.0
            
            # Forward pass
            activations = [X]
            logits_list = []
            current = X
            
            for layer_idx in range(len(weights)):
                logit = current @ weights[layer_idx] + biases[layer_idx]
                logits_list.append(logit)
                
                if layer_idx < len(weights) - 1:
                    output = sigmoid(logit)
                else:
                    output = softmax(logit)
                    
                activations.append(output)
                current = output
            
            # Backward pass
            delta = activations[-1] - y
                
            for layer_idx in reversed(range(len(weights))):
                a_prev = activations[layer_idx]
                
                # Average of the gradient
                dW = a_prev.T @ delta / batch_size
                # Average instead of reshape
                db = np.mean(delta, axis=0)
                
                weights[layer_idx] -= learning_rate * dW
                biases[layer_idx] -= learning_rate * db
                
                if layer_idx > 0:
                    delta = (delta @ weights[layer_idx].T) * sigmoid_derivative(activations[layer_idx])
            
            # Reset batch
            batch_images = []
            batch_labels = []
            
            if index % 5000 < batch_size:
                print(f"Processed {index} examples")

    epochEndTime = time.time()
    print(f"Time: {epochEndTime - epochStartTime:.3f} seconds")

totalEndTime = time.time()
print(f"Total time: {totalEndTime - totalStartTime:.3f} seconds ")

# Evaluation on test set
correct = 0
total = 0
for example in ds["test"]:
    total += 1

    # Forward pass only (same logic as training forward pass), just logits and activations not needed
    inputValue = (np.array(example["image"]).flatten() / 255.0).reshape(1, 784)
    for layer_idx in range(len(weights)):
        logit = inputValue @ weights[layer_idx] + biases[layer_idx]
        if layer_idx < len(weights) - 1:
            inputValue = sigmoid(logit)
        else:
            inputValue = softmax(logit)

    # Predicted label is the largest of the softmax output
    predicted_label = int(np.argmax(inputValue, axis=-1)[0])
    actual_label = int(example["label"])

    # Update counters
    if predicted_label == actual_label:
        correct += 1

# Save the weights and biases in .npz format
np.savez(
    "mnist_model.npz",
    weights=np.array(weights, dtype=object),
    biases=np.array(biases, dtype=object)
)

# Save the weights and biases in .json format
model_data = {
    "weights": [w.tolist() for w in weights],
    "biases": [b.tolist() for b in biases]
}
with open("model.json", "w") as f:
    json.dump(model_data, f)

# Final percentages
percent_correct = (correct / total) * 100.0 if total else 0.0
percent_incorrect = 100.0 - percent_correct

print(f"Test results: {correct}/{total} correct")
print(f"Percentage correct: {percent_correct:.2f}%")
print(f"Percentage incorrect: {percent_incorrect:.2f}%")