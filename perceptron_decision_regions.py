
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0

    def predict(self, x):
        summation = np.dot(self.weights, x) + self.bias
        return 1 if summation >= 0 else 0

    def train(self, x, y, learning_rate=1, epochs=100):
        for i in range(epochs):
            for xi, target in zip(x,y):
                prediction = self.predict(xi)
                self.weights += learning_rate * (target - prediction) * xi
                self.bias += learning_rate * (target - prediction)

    def plot_decision_regions(self, x, y):
        x_min = x[:, 0].min() - 1
        x_max = x[:, 0].max() + 1
        y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        Z = np.array([self.predict([x1, x2]) for x1, x2 in np.c_[xx.ravel(), yy.ravel()]])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.5)
        plt.scatter(x[:, 0], x[:, 1], c=y, s=20, edgecolor='k')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Perceptron Decision Regions')
        plt.show()


X_train, y_train = make_classification(n_samples=100, n_features=2,
                           n_redundant=0, n_clusters_per_class=1,
                           class_sep=2)

perceptron = Perceptron(input_size=2)

# Train the perceptron
perceptron.train(X_train, y_train)
perceptron.plot_decision_regions(X_train, y_train)