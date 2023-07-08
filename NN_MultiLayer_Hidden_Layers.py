#!/usr/bin/env python
# coding: utf-8

# In[29]:


# Question 7 [1.5 pts] 
# Please revise the MultiLayer NN Decision Boundary Notebook in the Canvas to implement following tasks


# Importing needed libraries for executing tasks
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier





# In[30]:


#trial

X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(3,), random_state=42)
model.fit(X_train, y_train)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr')
plt.title('Moon-shaped Dataset - Training Instances')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


# In[31]:


# 7.1 
# Generate a moon shaped dataset with two class lables (including 0.3 noise). Split the data into 60% training and 40% test data. 
# Use 60% training data to train a neural network with 3 hidden nodes. Show the training instances on the plot 
# (color node into different colors, depending on the lables).


# Step 1: Generate a moon-shaped dataset with two class labels
X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)

# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Step 3: Train a neural network with 3 hidden nodes
model_3 = MLPClassifier(hidden_layer_sizes=(3,), random_state=42)
model_3.fit(X_train, y_train)

# Step 4: Plot training instances
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolors='k')

# Step 5: Plot lines corresponding to hidden nodes
for i, coef in enumerate(model_3.coefs_[0]):
    line = np.linspace(-2, 3, 100)
    plt.plot(line, -(coef[0] * line + model_3.intercepts_[0][i]) / coef[1])

# Additional settings for plot visualization
plt.xlim(-2, 3)
plt.ylim(-2, 2)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Decision Boundary - 3 Hidden Nodes")
plt.show()



# In[32]:


# 7.2 
# Retrain the network using 4 hidden nodes, and 5 hidden nodes, respectively. Show the four lines corresponding to the
# four hidden nodes, and the five lines corresponding to the five hidden nodes, respectively, on two separated plots 
# (show the training instances on the same plot (color node into different colors, depending on the labels)) [0.5 pt]



# Generate a moon-shaped dataset with two class labels and 0.3 noise
X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train a neural network with 4 hidden nodes
hidden_nodes_4 = 4
classifier_4 = MLPClassifier(hidden_layer_sizes=(hidden_nodes_4,), random_state=42)
classifier_4.fit(X_train, y_train)

# Train a neural network with 5 hidden nodes
hidden_nodes_5 = 5
classifier_5 = MLPClassifier(hidden_layer_sizes=(hidden_nodes_5,), random_state=42)
classifier_5.fit(X_train, y_train)

# Plot the training instances with different colors for each label and lines for hidden nodes (4 hidden nodes)
plt.figure(figsize=(12, 4))

plt.subplot(121)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolors='k')

weights_4 = classifier_4.coefs_[0]
biases_4 = classifier_4.intercepts_[0]
for i in range(hidden_nodes_4):
    line_x = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]), 100)
    line_y = -(weights_4[0, i] * line_x + biases_4[i]) / weights_4[1, i]
    plt.plot(line_x, line_y, label=f'Hidden Node {i+1}')

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Decision Boundary with 4 Hidden Nodes')

# Plot the training instances with different colors for each label and lines for hidden nodes (5 hidden nodes)
plt.subplot(122)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolors='k')

weights_5 = classifier_5.coefs_[0]
biases_5 = classifier_5.intercepts_[0]
for i in range(hidden_nodes_5):
    line_x = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 0]), 100)
    line_y = -(weights_5[0, i] * line_x + biases_5[i]) / weights_5[1, i]
    plt.plot(line_x, line_y, label=f'Hidden Node {i+1}')

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Decision Boundary with 5 Hidden Nodes')

plt.tight_layout()
plt.show()


# In[33]:


# trial

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolors='k')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Decision Boundaries with 3 Hidden Nodes")

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),np.arange(y_min, y_max, 0.01))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
plt.show()


# In[34]:


# with 4 and 5 hidden nodes

# Generate a moon-shaped dataset with two class labels and 0.3 noise
X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)

# Split the data into 60% training and 40% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train a neural network with 4 hidden nodes
model_4 = MLPClassifier(hidden_layer_sizes=(4,), random_state=42)
model_4.fit(X_train, y_train)

# Train a neural network with 5 hidden nodes
model_5 = MLPClassifier(hidden_layer_sizes=(5,), random_state=42)
model_5.fit(X_train, y_train)

# Visualize the training instances and decision boundaries for 4 hidden nodes
plt.figure(figsize=(12, 4))

# Plot for 4 hidden nodes
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolors='k')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Decision Boundaries with 4 Hidden Nodes")

# Generate a meshgrid to plot the decision boundaries
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Make predictions on the meshgrid points
Z = model_4.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

# Visualize the training instances and decision boundaries for 5 hidden nodes
# Plot for 5 hidden nodes
plt.subplot(1, 2, 2)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolors='k')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Decision Boundaries with 5 Hidden Nodes")

# Make predictions on the meshgrid points
Z = model_5.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries
plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')

plt.tight_layout()
plt.show()


# In[35]:


# 7.3 
# Explain how does the neural network decision surface change, when increasing the number of hidden nodes from 3, to 4, to 5. 
# Explain what are the benefits and risk of increasing number of hidden nodes, respectively [0.5 pt]



# Generate a moon-shaped dataset with two class labels and 0.3 noise
X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)

# Split the data into 60% training and 40% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Define the number of hidden nodes to experiment with
hidden_nodes = [3, 4, 5]

# Train and visualize the decision surface for each number of hidden nodes
for n_nodes in hidden_nodes:
    # Train a neural network with the specified number of hidden nodes
    model = MLPClassifier(hidden_layer_sizes=(n_nodes,), random_state=42)
    model.fit(X_train, y_train)

    # Visualize the decision surface
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolors='k')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Decision Surface with {} Hidden Nodes".format(n_nodes))

    # Generate a meshgrid to plot the decision surface
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Make predictions on the meshgrid points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision surface
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.show()

    # writing the benefits and risks
    print("Number of Hidden Nodes: {}".format(n_nodes))
    print("Benefits:")
    print("- The decision surface becomes more flexible and can capture complex patterns.")
    print("- Improved representation of intricate and nonlinear relationships in the data.")
    print("- Increased modeling capability for complex datasets with intricate decision boundaries.")
    print("Risks:")
    print("- Overfitting if the number of hidden nodes is excessively large relative to the problem complexity or training data size.")
    print("- Increased computational complexity and resource requirements for training and inference.")
    print("- Gradient instability and difficulties in training with deeper networks.")
    print("-----------------------------------------------------\n")


# In[ ]:




