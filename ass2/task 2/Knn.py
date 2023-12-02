import numpy as np
from collections import Counter



class Knn:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def ecludianDistance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, points):
        predictions = [self._predict(point) for point in points]
        return predictions
    
    def _predict(self, point): 
        distances = [self.ecludianDistance(point, i) for i in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        sorted_distances = [self.y_train[i] for i in k_indices]
        most_common = Counter(sorted_distances).most_common()
        return most_common[0][0]

