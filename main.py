import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def data_generation(n):
    def generate_point(colour, x_range, y_range):
        while True:
            x = random.randint(*x_range)
            y = random.randint(*y_range)
            if (x, y) not in x_y_coordinates:
                x_y_coordinates.add((x, y))
                return x, y, colour

    data = []
    x_y_coordinates = set()

    ranges = {
        'red': ([-5000, 500] if random.random() < 0.99 else [-5000, 5000], [-5000, 500] if random.random() < 0.99 else [-5000, 5000]),
        'green': ([-500, 5000] if random.random() < 0.99 else [-5000, 5000], [-5000, 500] if random.random() < 0.99 else [-5000, 5000]),
        'blue': ([-5000, 500] if random.random() < 0.99 else [-5000, 5000], [-500, 5000] if random.random() < 0.99 else [-5000, 5000]),
        'purple': ([-500, 5000] if random.random() < 0.99 else [-5000, 5000], [-500, 5000] if random.random() < 0.99 else [-5000, 5000])
    }

    for _ in range(n // 4):
        for colour in ('red', 'green', 'blue', 'purple'):
            x_range, y_range = ranges[colour]
            data.append(generate_point(colour, x_range, y_range))

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

train_data = [
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
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


test_data = data_generation(40000)

coordinates_tensor_train, labels_tensor_train, _ = data_transform(train_data)
coordinates_tensor_test, labels_tensor_test, coordinates_test = data_transform(test_data)

model = ColorClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(coordinates_tensor_train)
    loss = criterion(outputs, labels_tensor_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    predicted = model(coordinates_tensor_test)
    _, predicted_labels = torch.max(predicted, 1)


predicted_labels = [label_map_inverse[label] for label in predicted_labels.tolist()]

plt.figure(figsize=(8, 6))
plt.scatter([x for x, _ in coordinates_test], [y for _, y in coordinates_test], c=predicted_labels)
plt.title('Predicted Clusters')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.show()
