#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Question 8 [1.5 pts]
# The Multi-Layer NN Face Recognition Notebook in the Canvas shows detailed procedures about how to use Oivetti face
# dataset (from AT&T) to train Neural Network classifiers for face classification. The dataset in the Canvas also 
# provides “olivetti_faces.npy” and “olivetti_faces_target.npy”, which includes 400 faces in 40 classes 
# (40 different person). Please implement following face recognition task using Neural Networks.


# Import the necessary libraries:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import warnings


# In[6]:


X = np.load('olivetti_faces.npy')
y = np.load('olivetti_faces_target.npy')

# data (400 images, each 64x64)
print(X.shape)
# labels
print(y.shape)
print("\n")
print(y)


# In[7]:


# 8.1 
# Please show at least one face images for each class in the Oivetti face dataset100. Randomly split the dataset into
# 60% training and 40% test samples. Train a one-hidden layer neural network with 10 hidden nodes. 
# Report the classification accuracy of the classifier on the test set [0.5 pt]


warnings.filterwarnings("ignore")

# Loading the Olivetti face dataset and target dataset
X = np.load('olivetti_faces.npy')
y = np.load('olivetti_faces_target.npy')

# Reshape the input array to have two dimensions
n_samples = X.shape[0]
X = X.reshape((n_samples, -1))

# Display one face image for each class
num_classes = np.unique(y)
fig, axes = plt.subplots(nrows=5, ncols=8, figsize=(10, 6))
for i, ax in enumerate(axes.flatten()):
    img = X[y == num_classes[i]][0].reshape(64, 64)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(f'Person {num_classes[i]}')
plt.tight_layout()
plt.show()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=40)

# Train the neural network classifier
hidden_nodes = 10
classifier = MLPClassifier(hidden_layer_sizes=(hidden_nodes,), random_state=42)
classifier.fit(X_train, y_train)

# Predict the labels for the test set and calculate the classification accuracy
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification accuracy: {accuracy}")




# In[8]:


# Flatten each 64x64 image into a single vector
X = X.reshape((X.shape[0], -1))

print(X.shape)


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
print(X_train.shape)
print(X_test.shape)

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=500, random_state=42,activation='logistic',max_iter=1000)
clf.fit(X_train, y_train)


# In[10]:


y_pred=clf.predict(X_test)
print(y_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cf=confusion_matrix(y_test, y_pred)
cf


# In[11]:


# use scikit-learn to calculate accuracy. 
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


# In[12]:


# 8.2 
# Please use one time 10-fold cross validation to compare the performance of different neural network architectures, 
# including (1) one-hidden layer NN with 10 hidden nodes, (2) one-hidden layer NN with 50 hidden nodes, 
# (3) one-hidden layer NN with 500 hidden nodes, and (4) two-hidden layer NN with 50 hidden nodes (1st layer) and 
# 10 hidden nodes (2n layer). Please report and compare the cross-validation accuracy of the four neural networks, 
# and conclude which classifier has the best performance [1 pt].

from sklearn.model_selection import KFold

# Loading the Olivetti face dataset and target dataset
X = np.load('olivetti_faces.npy')
y = np.load('olivetti_faces_target.npy')

# Reshape the input array to have two dimensions
n_samples = X.shape[0]
X = X.reshape((n_samples, -1))

kf = KFold(n_splits=10,shuffle=True)
kf.get_n_splits(X)

Acc = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf1 = MLPClassifier(solver='lbfgs', hidden_layer_sizes=50, random_state=42, activation= 'logistic', max_iter=1000)
    clf1.fit(X_train, y_train)
    y_pred = clf1.predict(X_test)
    Acc.append(accuracy_score(y_test, y_pred))
    
print(Acc)
print("The average accuracy of the Classifier is %.4f" % np.mean(Acc))


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import warnings


warnings.filterwarnings("ignore")

X = np.load('olivetti_faces.npy')
y = np.load('olivetti_faces_target.npy')

X = X.reshape((X.shape[0], -1))

kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(X)

architectures = [
    {'hidden_layer_sizes': (10,), 'name': 'One-hidden layer (10 nodes)'},
    {'hidden_layer_sizes': (50,), 'name': 'One-hidden layer (50 nodes)'},
    {'hidden_layer_sizes': (500,), 'name': 'One-hidden layer (500 nodes)'},
    {'hidden_layer_sizes': (50, 10), 'name': 'Two-hidden layer (50, 10 nodes)'}
]

results = []

for architecture in architectures:
    Acc = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = MLPClassifier(solver='lbfgs', random_state=42, activation='logistic',
                            max_iter=1000, hidden_layer_sizes=architecture['hidden_layer_sizes'])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        Acc.append(accuracy_score(y_test, y_pred))

    avg_acc = np.mean(Acc)
    results.append({'architecture': architecture['name'], 'accuracy': avg_acc})

# Print the results
for result in results:
    print(f"Architecture: {result['architecture']}")
    print(f"Cross-validation Accuracy: {result['accuracy']:.4f}")
    print('')

best_architecture = max(results, key=lambda x: x['accuracy'])
print(f"The best performing architecture is: {best_architecture['architecture']}")


# In[ ]:




