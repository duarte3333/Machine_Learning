import matplotlib.pyplot as plt
import numpy as np

# Generate a random dataset with two classes
np.random.seed(0)
class_1 = np.random.rand(10, 2)  # Class 1 points
class_2 = np.random.rand(10, 2) + np.array([1.2, 0.8])  # Class 2 points

# Create a combined dataset
X = np.vstack((class_1, class_2))

# Labels: 0 for class 1, 1 for class 2
y = np.hstack((np.zeros(10), np.ones(10)))

# Random query point
query_point = np.random.rand(1, 2)

# Calculate distances from the query point to each data point
distances = np.linalg.norm(X - query_point, axis=1)

# Find the nearest neighbor index
nearest_neighbor_index = np.argmin(distances)

# Predicted label based on nearest neighbor
predicted_label = y[nearest_neighbor_index]

# Calculate accuracy
accuracy = 1.0 if predicted_label == 1.0 else 0.0  # Binary classification

# Plot the data points and the query point
plt.scatter(class_1[:, 0], class_1[:, 1], label='Class 1')
plt.scatter(class_2[:, 0], class_2[:, 1], label='Class 2')
plt.scatter(query_point[:, 0], query_point[:, 1], color='red', label='Query Point')
plt.scatter(X[nearest_neighbor_index, 0], X[nearest_neighbor_index, 1], color='green', label='Nearest Neighbor')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Nearest Neighbor Classification')
plt.legend()
plt.show()

print('Predicted label for the query point:', predicted_label)
print('Accuracy:', accuracy)
