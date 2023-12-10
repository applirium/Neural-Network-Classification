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

    test_data = data_generation(40000, False)

    coordinates_tensor_train, labels_tensor_train, coordinates_train = data_transform(train_data)
    coordinates_tensor_test, labels_tensor_test, coordinates_test = data_transform(test_data)

    model = ColorClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_losses = []
    accuracies = []

    num_epochs = 50
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(coordinates_tensor_train)
        loss = criterion(outputs, labels_tensor_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())  # Store training loss for each epoch

        # Evaluation on test data
        model.eval()
        with torch.no_grad():
            predicted = model(coordinates_tensor_test)
            test_loss = criterion(predicted, labels_tensor_test)
            test_losses.append(test_loss.item())  # Store test loss for each epoch

            _, predicted_labels = torch.max(predicted.data, 1)
            accuracy = torch.sum(predicted_labels == labels_tensor_test).item() / len(labels_tensor_test)
            accuracies.append(accuracy * 100)  # Store accuracy for each epoch
            print(f"Epoch [{epoch + 1}/{num_epochs}] Training Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f} | Accuracy: {accuracy * 100:.2f}%")

    model.eval()
    with torch.no_grad():
        predicted = model(coordinates_tensor_test)
        _, predicted_labels = torch.max(predicted.data, 1)

    acc = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == labels_tensor_test[i]:
            acc += 1
    acc = (acc / len(predicted_labels)) * 100
    print(f"Accuracy of model: {acc:.2f}%")

    starting_labels = [label_map_inverse[label] for label in labels_tensor_train.tolist()]
    predicted_labels = [label_map_inverse[label] for label in predicted_labels.tolist()]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].scatter(
        [x for x, _ in coordinates_train],
        [y for _, y in coordinates_train],
        c=starting_labels,
    )
    axes[0, 0].set_xlabel('X-coordinate')
    axes[0, 0].set_ylabel('Y-coordinate')
    axes[0, 0].set_title(f'Training on {choice} elements')

    axes[0, 1].scatter(
        [x for x, _ in coordinates_test],
        [y for _, y in coordinates_test],
        c=predicted_labels,
    )
    axes[0, 1].set_xlabel('X-coordinate')
    axes[0, 1].set_ylabel('Y-coordinate')
    axes[0, 1].set_title(f'Testing on 40k elements')

    axes[1, 0].plot(train_losses, label='Training Loss')
    axes[1, 0].plot(test_losses, label='Test Loss')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training and Test Losses over Epochs')
    axes[1, 0].legend()

    axes[1, 1].plot(accuracies)
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Accuracy over Epochs')

    plt.show()
