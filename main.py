import scipy.io
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learn_rate, threshold=0):
        self.learn_rate = learn_rate
        self.threshold = threshold
        self.weights = np.zeros(2)
        self.bias = 0
        
    def activation(self, x):
    
        value = np.dot(x, self.weights) + self.bias
        
        if value > self.threshold:
            return 1
        if value < -self.threshold:
            return -1
        return 0
    
    def train(self, x, y):
        total_errors = 0
        for i in range(len(x)):
            prediction = self.activation(x[i])
            
            if prediction != y[i]:
                self.weights +=  self.learn_rate * y[i] * x[i]
                self.bias += self.learn_rate * y[i]
                total_errors += 1
                    
        return total_errors
    
    def train_to_no_error(self, x, y):
        errors = []
        epochs = 1
        error = self.train(x, y)
        
        print(f"Epoch: {epochs}, Error: {error}")
        
        while error:
            errors.append(error)
            epochs += 1
            error = self.train(x, y)
            
            print(f"Epoch: {epochs}, Error: {error}")
        return errors
    
    def train_to_n(self, n, x, y):
        errors = []
        for i in range(n):
            errors.append(self.train(x, y))
    
        return errors

learn_rate = 0.1

def plot_decision_boundary(perceptron, x, y, title):
    # Mesh grid to plot decision boundary
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Predict each point on the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([perceptron.activation(point) for point in grid_points])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary and points
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    plt.title(title)
    plt.xlabel('Class 1')
    plt.ylabel('Class 2')
    plt.show()
    
def compute_error_rate(perceptron, x, y):
    total_errors = 0
    for i in range(len(x)):
        prediction = perceptron.activation(x[i])
        if prediction != y[i]:
            total_errors += 1
            
    error_rate = total_errors / len(x)
    
    return error_rate


epochs = 35

errorsA = []
setA = scipy.io.loadmat("setA.math.mat")
perceptronA = Perceptron(learn_rate=0.2, threshold=0)
errorsA = perceptronA.train_to_n(epochs, setA["X"], setA["Y"])

errorsB = []
setB = scipy.io.loadmat("setB.math.mat")
perceptronB = Perceptron(learn_rate=0.4, threshold=0.1)
errorsB = perceptronB.train_to_n(epochs, setB["X"], setB["Y"])

plt.figure(figsize=(10, 6))


# Plots Training Error vs Epochs for A
plt.title("Training Error vs. Number of Epochs for Set A")
plt.xlabel("Epochs")
plt.ylabel("Number of Errors")
plt.plot(range(1, epochs+1), errorsA, label="Set A(Linearly Seperable)", color="blue")
plt.legend()
plt.show()


# Plots Training Error vs Epochs for B
plt.title("Training Error vs. Number of Epochs for Set B")
plt.xlabel("Epochs")
plt.ylabel("Number of Errors")
plt.plot(range(1, epochs + 1), errorsB, label="Set B (Non-Linearly Separable)", color="red")
plt.legend()
plt.show()


# Plots decision boundary and test data for A
plot_decision_boundary(perceptronA, setA['X'], setA['Y'], "Boundary Decision for Test Data A (Linearly Seperable)")

# Plots decision boundary and test data for B
plot_decision_boundary(perceptronB, setB['X'], setB['Y'], "Boundary Decision for Test Data B (Non-Linearly Seperable)")


# Compute error rate for Set A
error_rate_A = compute_error_rate(perceptronA, setA['X'], setA['Y'])
print(f"Overall error rate for Set A: {error_rate_A * 100:.2f}%")


# Compute error rate for Set B
error_rate_B = compute_error_rate(perceptronB, setB['X'], setB['Y'])
print(f"Overall error rate for Set B: {error_rate_B * 100:.2f}%")
