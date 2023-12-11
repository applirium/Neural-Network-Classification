import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


label_map = {'red': 0, 'green': 1, 'blue': 2, 'purple': 3}              # Mapping of labels to numerical values and vice versa
label_map_inverse = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple'}

start_data = [                                                          # Initial dataset containing coordinates and colors
    (-4500, -4400, 'red'), (-4100, -3000, 'red'), (-1800, -2400, 'red'), (-2500, -3400, 'red'), (-2000, -1400, 'red'),
    (4500, -4400, 'green'), (4100, -3000, 'green'), (1800, -2400, 'green'), (2500, -3400, 'green'), (2000, -1400, 'green'),
    (-4500, 4400, 'blue'), (-4100, 3000, 'blue'), (-1800, 2400, 'blue'), (-2500, 3400, 'blue'), (-2000, 1400, 'blue'),
    (4500, 4400, 'purple'), (4100, 3000, 'purple'), (1800, 2400, 'purple'), (2500, 3400, 'purple'), (2000, 1400, 'purple')
]


def data_generation(n, train):                          # Function to generate new data points for the dataset
    def generate_point(x_range, y_range):               # Generate random coordinates within given ranges
        while True:
            x = random.randint(*x_range)
            y = random.randint(*y_range)
            if (x, y) not in x_y_coordinates:
                x_y_coordinates.add((x, y))
                return x, y, colour

    data = []
    x_y_coordinates = set()

    if train:                                           # Define ranges for generating data points based on 'train' flag
        ranges = {
            'red': ([-5000, 0], [-5000, 0]),
            'green': ([0, 5000], [-5000, 0]),
            'blue': ([-5000, 0], [0, 5000]),
            'purple': ([0, 5000], [0, 5000])
        }

    else:
        ranges = {
            'red': ([-5000, 500] if random.random() < 0.99 else [-5000, 5000], [-5000, 500] if random.random() < 0.99 else [-5000, 5000]),
            'green': ([-500, 5000] if random.random() < 0.99 else [-5000, 5000], [-5000, 500] if random.random() < 0.99 else [-5000, 5000]),
            'blue': ([-5000, 500] if random.random() < 0.99 else [-5000, 5000], [-500, 5000] if random.random() < 0.99 else [-5000, 5000]),
            'purple': ([-500, 5000] if random.random() < 0.99 else [-5000, 5000], [-500, 5000] if random.random() < 0.99 else [-5000, 5000])
        }

    for _ in range(n // 4):                             # Generate new data points based on color ranges
        for colour in ('red', 'green', 'blue', 'purple'):
            x_range, y_range = ranges[colour]
            data.append(generate_point(x_range, y_range))

    return data


def data_transform(input_data):                         # Function to transform data into PyTorch tensors
    coordinates = [(x, y) for x, y, _ in input_data]
    labels = [label for _, _, label in input_data]      # Extract coordinates and labels from input data
    labels = [label_map[label] for label in labels]

    # Convert data to PyTorch tensors
    coordinates_tensor = torch.tensor(coordinates, dtype=torch.float32)     # Convert data to PyTorch tensors
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return coordinates_tensor, labels_tensor, coordinates


class ColorClassifier(nn.Module):
    def __init__(self):                                  # Define the neural network layers
        super(ColorClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 64),       # Input layer: 2 input features, 64 output features
            nn.ReLU(),                                   # Activation function: ReLU
            nn.Linear(64, 64),      # Hidden layer 1: 64 input features, 64 output features
            nn.ReLU(),                                   # Activation function: ReLU
            nn.Linear(64, 64),      # Hidden layer 2: 64 input features, 64 output features
            nn.ReLU(),                                   # Activation function: ReLU
            nn.Linear(64, 4)         # Output layer: 64 input features, 4 output features (classes)
        )

    def forward(self, x):                               # Forward pass through the network
        x = self.fc(x)
        return x


def test():
    while True:
        while True:
            choice = int(input("Zadaj počet trénovacích bodov: "))              # Input choice for the number of training points
            if choice < 20:
                print("Min = 20 bodov")
            else:
                break

        if choice == 20:                                                        # Generate or select training and test data based on the user's choice
            train_data = start_data                                             # Use initial dataset
        else:
            train_data = start_data + data_generation(choice - 20, True)   # Generate additional training data

        test_data = data_generation(20000, False)                       # Generate test data (20000 points)

        # Transform data into PyTorch tensors
        coordinates_tensor_train, labels_tensor_train, coordinates_train = data_transform(train_data)
        coordinates_tensor_test, labels_tensor_test, coordinates_test = data_transform(test_data)

        # Initialize the model, loss function, and optimizer
        model = ColorClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Lists to store training and test losses, and accuracies
        train_losses = []
        test_losses = []
        accuracies_train = []
        accuracies_test = []
        prediction = []         # List to store predicted labels

        num_epochs = 50         # Number of training epochs
        model.train()           # Set model to training mode
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(coordinates_tensor_train)
            loss = criterion(outputs, labels_tensor_train)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())            # Store training loss for each epoch

            # Calculate training accuracy for the current epoch
            _, predicted_train_labels = torch.max(outputs.data, 1)
            accuracy_train = torch.sum(predicted_train_labels == labels_tensor_train).item() / len(labels_tensor_train)
            accuracies_train.append(accuracy_train * 100)

            # Model evaluation on test data
            model.eval()                                # Set model to evaluation mode
            with torch.no_grad():
                predicted = model(coordinates_tensor_test)
                test_loss = criterion(predicted, labels_tensor_test)
                test_losses.append(test_loss.item())    # Store test loss for each epoch

                _, predicted_test_labels = torch.max(predicted.data, 1)
                accuracy_test = torch.sum(predicted_test_labels == labels_tensor_test).item() / len(labels_tensor_test)
                accuracies_test.append(accuracy_test * 100)
                prediction.append([label_map_inverse[label] for label in predicted_test_labels.tolist()])

                # Display training and test statistics for each epoch
                print(f"Epoch [{epoch + 1}/{num_epochs}] Training Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f} | Accuracy: {accuracy_train * 100:.2f}% | Accuracy: {accuracy_test * 100:.2f}%")

        starting_labels = [label_map_inverse[label] for label in labels_tensor_train.tolist()]

        plt.ion()                                           # Turn on interactive mode for live plotting
        plt.subplots(2, 2, figsize=(10, 10))    # Create subplots for visualizing the training process

        for i in range(num_epochs):                         # Loop through each epoch
            plt.clf()                                       # Clear the current figure

            plt.subplot(2, 2, 1)                      # Subplot 1: Training data scatter plot
            plt.scatter([x for x, _ in coordinates_train], [y for _, y in coordinates_train], c=starting_labels)

            plt.xlabel('X-coordinate')
            plt.ylabel('Y-coordinate')
            plt.title(f'Training on {choice} elements')

            plt.subplot(2, 2, 2)                       # Subplot 2: Test data scatter plot with predicted labels for current epoch
            plt.scatter([x for x, _ in coordinates_test], [y for _, y in coordinates_test], c=prediction[i])

            plt.xlabel('X-coordinate')
            plt.ylabel('Y-coordinate')
            plt.title(f'Epoch {i + 1} Testing ')

            plt.subplot(2, 2, 3)                       # Subplot 3: Training and test losses over epochs
            plt.plot(train_losses[0:i], label='Training Loss')
            plt.plot(test_losses[0:i], label='Test Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Training Loss: {round(train_losses[i],2)}\nTest Loss: {round(test_losses[i],2)}')
            plt.legend()

            plt.subplot(2, 2, 4)                       # Subplot 4: Training and test accuracies over epochs
            plt.plot(accuracies_train[0:i], label='Training Accuracy')
            plt.plot(accuracies_test[0:i], label='Test Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.title(f'Training Accuracy: {round(accuracies_train[i],2)}\nTest Accuracy: {round(accuracies_test[i],2)}')
            plt.legend()

            plt.tight_layout()          # Adjust layout for better appearance
            plt.pause(0.01)             # Pause to allow the plot to be updated
            plt.draw()                  # Draw the updated plot

        plt.pause(5)                    # Pause for 5 seconds before closing the plot
        plt.close()                     # Close the plot window


test()
