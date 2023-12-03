import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
from Knn import Knn

# Read the CSV file
data = pd.read_csv('ass2/task 2/diabetes.csv')

# MinMaxScalar
def minMaxScaling(col):
    min_X = col.min()
    max_X = col.max()
    scaled_X = (col - min_X) / (max_X - min_X)
    return scaled_X



# Drop Outcome column from dataset
x = data.drop(columns=["Outcome"])

# Apply Normalization
x = np.apply_along_axis(minMaxScaling, axis=0, arr=x)

y = data["Outcome"].to_numpy()

# Split the dataset to training set and testing set
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=.3, random_state=1234)


# Apply Knn to the normalized training set
knn = Knn()
knn.fit(X_train, Y_train)

Y_predict = knn.predict(X_test)

print("k value: 3")
# calculate the number of correctly classified instances
correctly_classified_instances = np.sum(Y_predict == Y_test)
print ("The number of correctly classified instances: ",correctly_classified_instances)

# Number of total instances
total_instances = len(Y_test)
print ("The number of total instances: ", total_instances)

# Calculate the accuracy
acc = correctly_classified_instances / total_instances * 100
print("Accuracy:", acc, '%')



