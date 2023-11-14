import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Reading Files
data = pd.read_csv("loan_old.csv")
# filterizating and preprocessing
missing_values = data.isnull().sum()
data = data.dropna()
data = data.drop(columns=['Loan_ID','Max_Loan_Amount'],axis=1)
x = data.drop(columns=['Loan_Status'])
y = data['Loan_Status']
y = y.map({'Y':1,'N':0})
x = pd.get_dummies(x, columns=["Gender", "Married", "Dependents", "Education", "Credit_History", "Property_Area"], drop_first=True)

x_train,x_test,y_train,y_test = train_test_split(
    x.values,y.values,test_size=0.3,random_state=42
)

# test
print(x_train.mean(),"\n\n", x_test)

# Logistic model

def sigmoid(x):
    x = np.float64(x)
    return (1/(1+np.exp(-1*x)))

class LogisticRegression:

    def __init__(self, lr = 0.01, iterations = 10):
        self.learning_rate = lr
        self.iterations = iterations
        self.thetas = None
        self.theta0 = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        m_sample, n_features = X.shape
        self.thetas = np.zeros(n_features)
        self.theta0 = 0

        for _ in range(self.iterations):
            linear_prediction = np.dot(X, self.thetas) + self.theta0
            prediction = sigmoid(linear_prediction)

            d_thetas = (1 / m_sample) * np.dot(X.T, (prediction - y))
            d_theta0 = (1 / m_sample) * np.sum(prediction - y)
            
            self.thetas -= self.learning_rate * d_thetas.astype(float)
            self.theta0 -= self.learning_rate * d_theta0

    def predict(self, X):
        linear_pred = np.dot(X, self.thetas) + self.theta0
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y < 0.5 else 1 for y in y_pred]
        return class_pred
    
#Main
lr = LogisticRegression()
lr.fit(x_train,y_train)
print(lr.thetas)
print(lr.theta0)
y_predict = lr.predict(x_test)

def acc(y_predict,y_real):
    return np.sum(y_predict == y_real)/len(y_real)

acc = acc(y_predict,y_test)
print("accuracy of my model is ",acc*100)