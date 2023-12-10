import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def data_generation(n, train):
    def generate_point(x_range, y_range):
        while True:
            x = random.randint(*x_range)
            y = random.randint(*y_range)
            if (x, y) not in x_y_coordinates:
                x_y_coordinates.add((x, y))
                return x, y, colour

    data = []
    x_y_coordinates = set()

    if train:
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

    for _ in range(n // 4):
        for colour in ('red', 'green', 'blue', 'purple'):
            x_range, y_range = ranges[colour]
            data.append(generate_point(x_range, y_range))

    return data


def data_transform(input_data):
    coordinates = [(x, y) for x, y, _ in input_data]
    labels = [label for _, _, label in input_data]
    labels = [label_map[label] for label in labels]

    # Convert data to PyTorch tensors
    coordinates_tensor = torch.tensor(coordinates, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return coordinates_tensor, labels_tensor, coordinates


label_map = {'red': 0, 'green': 1, 'blue': 2, 'purple': 3}
label_map_inverse = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple'}

start_data = [
    (-4500, -4400, 'red'), (-4100, -3000, 'red'), (-1800, -2400, 'red'), (-2500, -3400, 'red'), (-2000, -1400, 'red'),
    (4500, -4400, 'green'), (4100, -3000, 'green'), (1800, -2400, 'green'), (2500, -3400, 'green'), (2000, -1400, 'green'),
    (-4500, 4400, 'blue'), (-4100, 3000, 'blue'), (-1800, 2400, 'blue'), (-2500, 3400, 'blue'), (-2000, 1400, 'blue'),
    (4500, 4400, 'purple'), (4100, 3000, 'purple'), (1800, 2400, 'purple'), (2500, 3400, 'purple'), (2000, 1400, 'purple')
]


class ColorClassifier(nn.Module):
    def __init__(self):
        super(ColorClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


while True:
    while True:
        choice = int(input("Zadaj počet trénovacích bodov: "))
        if choice < 20:
            print("Min = 20 bodov")
        else:
            break

    if choice == 20:
        train_data = start_data
    else:
        train_data = start_data + data_generation(choice - 20, True)

    test_data = data_generation(20000, False)

    coordinates_tensor_train, labels_tensor_train, coordinates_train = data_transform(train_data)
    coordinates_tensor_test, labels_tensor_test, coordinates_test = data_transform(test_data)

    model = ColorClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_losses = []
    accuracies_train = []
    accuracies_test = []
    prediction = []

    num_epochs = 50
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(coordinates_tensor_train)
        loss = criterion(outputs, labels_tensor_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())  # Store training loss for each epoch

        _, predicted_train_labels = torch.max(outputs.data, 1)
        accuracy_train = torch.sum(predicted_train_labels == labels_tensor_train).item() / len(labels_tensor_train)
        accuracies_train.append(accuracy_train * 100)

        # Evaluation on test data
        model.eval()
        with torch.no_grad():
            predicted = model(coordinates_tensor_test)
            test_loss = criterion(predicted, labels_tensor_test)
            test_losses.append(test_loss.item())  # Store test loss for each epoch

            _, predicted_test_labels = torch.max(predicted.data, 1)
            accuracy_test = torch.sum(predicted_test_labels == labels_tensor_test).item() / len(labels_tensor_test)
            accuracies_test.append(accuracy_test * 100)
            prediction.append([label_map_inverse[label] for label in predicted_test_labels.tolist()])

            print(f"Epoch [{epoch + 1}/{num_epochs}] Training Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f} | Accuracy: {accuracy_train * 100:.2f}% | Accuracy: {accuracy_test * 100:.2f}%")

    starting_labels = [label_map_inverse[label] for label in labels_tensor_train.tolist()]
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for i in range(num_epochs):
        plt.clf()

        plt.subplot(2, 2, 1)
        plt.scatter([x for x, _ in coordinates_train], [y for _, y in coordinates_train], c=starting_labels)

        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.title(f'Training on {choice} elements')

        plt.subplot(2, 2, 2)
        plt.scatter([x for x, _ in coordinates_test], [y for _, y in coordinates_test], c=prediction[i])

        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.title(f'Epoch {i + 1} Testing ')

        plt.subplot(2, 2, 3)
        plt.plot(train_losses[0:i], label='Training Loss')
        plt.plot(test_losses[0:i], label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Test Losses over Epochs')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(accuracies_train[0:i], label='Training Accuracy')
        plt.plot(accuracies_test[0:i], label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Training and Test Accuracies over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.pause(0.01)
        plt.draw()

    plt.pause(5)
    plt.close()
