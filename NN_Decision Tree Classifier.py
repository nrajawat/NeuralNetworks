#!/usr/bin/env python
# coding: utf-8

# In[60]:


#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, make_scorer, roc_auc_score
from sklearn.tree import export_graphviz, plot_tree
from sklearn.model_selection import cross_val_score


# In[70]:


data = pd.read_csv('housing.header.binary.txt')
data


# In[61]:


# [7.1]
# Use “Crim” and “Rm” as independent variables to train a decision tree to predict whether a house value 
# Medv is 1 or 0 (i.e., whether the house value is greater than 230k or not). [0.5 pt]

# Load the dataset into a pandas DataFrame
data = pd.read_csv('housing.header.binary.txt')

# dataset into independent (X) and the target  (y) variable
X = data[['Crim', 'Rm']]
y = data['Medv']

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

result = pd.concat([X, y], axis=1)
print(result)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
classifier = DecisionTreeClassifier()

# Fit the classifier to the training data
classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(" \n Accuracy Value is:", accuracy)


# In[62]:


#[7.2]
#Visualize the tree (including nodes and labels) and explain the meaning of the values showing in the root node [0.5 pt].

# Create and train the decision tree classifier
tree = DecisionTreeClassifier()
tree.fit(X, y)

# Visualize the tree
plt.figure(figsize=(10, 6))
plot_tree(tree, filled=True, feature_names=X.columns, class_names=['0', '1'])
plt.show()

print("""

In this, X represents the input features and y represents the target variable. We need to replace X and y with the actual data.

The plot_tree function plots the decision tree using matplotlib. The filled=True parameter fills the tree nodes with colors based on the majority class, and the feature_names and class_names parameters provide labels for the features and classes, respectively.

The root node represents the starting point of the decision tree. The values in the root node correspond to the splitting criterion used at that node. For example, if the root node has a splitting criterion of "RM <= 6.94", it means that the decision tree is splitting the data based on the feature "RM" and the threshold value of 6.94. The splitting criterion in the root node helps in determining how the decision tree divides the data into different branches based on the feature values. Each subsequent node represents a decision based on a specific feature and threshold value, leading to further splitting or classification. By visualizing the tree and examining the values in the root node, we can gain insights into the initial splitting decision made by the decision tree and how it partitions the data based on the chosen feature and threshold.
""")


# In[63]:


# [ 7.3 ]
#Use 80% of instances in the “housing.header.binary.txt” dataset to build a decision tree classifier 
# (using all features) to predict house value Medv [0.5 pt]. 


# Load the dataset into a pandas DataFrame
data = pd.read_csv('housing.header.binary.txt')

# Split the dataset into independent variables (X) and the target variable (y)
X = data.drop('Medv', axis=1)  # All features except Medv
y = data['Medv']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
classifier = DecisionTreeClassifier()

# Fit the classifier to the training data
classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test)

y_pred


# In[64]:


# [7.4] Report the performance of the classifier on the remaining 20% of instances in the “housing.header.binary.txt”
#a.	Report confusion table, TPR, FPR, and the Accuracy [0.5 pt]
#b.	Report the ROC curve [0.25 pt]
#c.	Report the AUC value [0.25 pt]



# Load the dataset into a pandas DataFrame
data = pd.read_csv('housing.header.binary.txt')

# Split the dataset into independent variables (X) and the target variable (y)
X = data.drop('Medv', axis=1)  # All features except Medv
y = data['Medv']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the decision tree classifier
classifier = DecisionTreeClassifier()

# Fit the classifier to the training data
classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test)

print("[a]")
# Compute the confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")

print(confusion)

# Calculate the true positive rate (TPR) and false positive rate (FPR)
tn, fp, fn, tp = confusion.ravel()
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
print("TPR:", tpr)
print("FPR:", fpr)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\n [b]")
# Compute the ROC curve
y_scores = classifier.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

print("\n[c]")
# Calculate the AUC value
auc_value = auc(fpr, tpr)
print("AUC value:", auc_value)


# In[65]:


# [7.5] 
# Use 5-fold cross-validation to compare decision trees, trained using different parameter settings:

# I.Inside each fold of the 5-fold cross validation, using training set (and all features, excluding the class label) 
# to train three trees, by setting minimum number of samples allowed in each node as 1, 3, 5 respectively 
# (in Scikit learn, this is controlled by min_samples_split) [0.5 pt]


# Load the dataset into a pandas DataFrame
data = pd.read_csv('housing.header.binary.txt')

# Split the dataset into independent variables (X) and the target variable (y)
X = data.drop('Medv', axis=1)  # All features except Medv
y = data['Medv']

# Initialize decision tree classifiers with different min_samples_split values
tree1 = DecisionTreeClassifier(min_samples_split=2)
tree2 = DecisionTreeClassifier(min_samples_split=3)
tree3 = DecisionTreeClassifier(min_samples_split=5)

# Perform 5-fold cross-validation and evaluate each tree's performance
scores_tree1 = cross_val_score(tree1, X, y, cv=5)
scores_tree2 = cross_val_score(tree2, X, y, cv=5)
scores_tree3 = cross_val_score(tree3, X, y, cv=5)

print("I.")
# Print the average accuracy for each tree
print("Average Accuracy (min_samples_split=2):", scores_tree1.mean())
print("Average Accuracy (min_samples_split=3):", scores_tree2.mean())
print("Average Accuracy (min_samples_split=5):", scores_tree3.mean())


# In[66]:


# II.
# Record performance of each tree (in terms of Accuracy and AUC values) on the test set as a list. [0.25 pt]
 

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate each tree on the test set
tree1.fit(X_train, y_train)
y_pred_tree1 = tree1.predict(X_test)
accuracy_tree1 = accuracy_score(y_test, y_pred_tree1)
auc_tree1 = roc_auc_score(y_test, y_pred_tree1)

tree2.fit(X_train, y_train)
y_pred_tree2 = tree2.predict(X_test)
accuracy_tree2 = accuracy_score(y_test, y_pred_tree2)
auc_tree2 = roc_auc_score(y_test, y_pred_tree2)

tree3.fit(X_train, y_train)
y_pred_tree3 = tree3.predict(X_test)
accuracy_tree3 = accuracy_score(y_test, y_pred_tree3)
auc_tree3 = roc_auc_score(y_test, y_pred_tree3)

print("II.")
# Record the performance of each tree in a list
performance = [
    {'Tree': 'Tree1', 'Accuracy': accuracy_tree1, 'AUC': auc_tree1},
    {'Tree': 'Tree2', 'Accuracy': accuracy_tree2, 'AUC': auc_tree2},
    {'Tree': 'Tree3', 'Accuracy': accuracy_tree3, 'AUC': auc_tree3}
]

# Print the performance of each tree
for p in performance:
    print(f"{p['Tree']}: Accuracy = {p['Accuracy']:.4f}, AUC = {p['AUC']:.4f}")


# In[67]:


# III.Calculate mean accuracy, and mean AUC values for three trees, after the 5-fold cross validation. 
# Report means accuracy and mean AUC values [0.25 pt]


# Initialize decision tree classifiers
tree1 = DecisionTreeClassifier()
tree2 = DecisionTreeClassifier()
tree3 = DecisionTreeClassifier()

# Perform 5-fold cross-validation
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'auc': make_scorer(roc_auc_score)
}

accuracy_scores_tree1 = cross_val_score(tree1, X, y, cv=5, scoring='accuracy')
auc_scores_tree1 = cross_val_score(tree1, X, y, cv=5, scoring='roc_auc')

accuracy_scores_tree2 = cross_val_score(tree2, X, y, cv=5, scoring='accuracy')
auc_scores_tree2 = cross_val_score(tree2, X, y, cv=5, scoring='roc_auc')

accuracy_scores_tree3 = cross_val_score(tree3, X, y, cv=5, scoring='accuracy')
auc_scores_tree3 = cross_val_score(tree3, X, y, cv=5, scoring='roc_auc')

# Calculate mean accuracy and mean AUC values
mean_accuracy_tree1 = accuracy_scores_tree1.mean()
mean_auc_tree1 = auc_scores_tree1.mean()

mean_accuracy_tree2 = accuracy_scores_tree2.mean()
mean_auc_tree2 = auc_scores_tree2.mean()

mean_accuracy_tree3 = accuracy_scores_tree3.mean()
mean_auc_tree3 = auc_scores_tree3.mean()

print("III.")
# Print the mean accuracy and mean AUC values for each tree
print(f"Tree1: Mean Accuracy = {mean_accuracy_tree1:.4f}, Mean AUC = {mean_auc_tree1:.4f}")
print(f"Tree2: Mean Accuracy = {mean_accuracy_tree2:.4f}, Mean AUC = {mean_auc_tree2:.4f}")
print(f"Tree3: Mean Accuracy = {mean_accuracy_tree3:.4f}, Mean AUC = {mean_auc_tree3:.4f}")


# In[68]:


#IV
#Analysis the performance of the three trees and explain how does the minimum number of samples allowed 
#for each node (i.e., min_samples_split) impact on the tree performance. [0.5 pt]

print("IV.")

print("""
Explanation of how the minimum number of samples allowed for each node (i.e., min_samples_split) impacts the performance of the three trees:

The minimum number of samples allowed for each node, controlled by the min_samples_split parameter, has a significant impact on the performance of decision trees. When this value is set too low, the tree tends to overfit the training data by capturing noise and specific details of the training set. As a result, the tree may not generalize well to unseen data, leading to lower accuracy and AUC values on the test set.

On the other hand, setting the min_samples_split value too high can result in underfitting, where the tree fails to capture important patterns and relationships in the data. The tree becomes too generalized, leading to poor performance on both the training and test sets.

Choosing the optimal min_samples_split value is crucial for achieving a balance between capturing relevant information from the data and avoiding overfitting or underfitting. By tuning this parameter carefully, we can optimize the tree's performance on unseen data. It is recommended to perform a systematic search or use techniques like cross-validation to find the best min_samples_split value that maximizes the accuracy and AUC metrics on the test set.

Overall, the min_samples_split parameter acts as a regularization mechanism in decision trees, controlling the trade-off between model complexity and generalization. Finding the right balance is essential for building decision trees that perform well on unseen data."
""")


# In[ ]:




