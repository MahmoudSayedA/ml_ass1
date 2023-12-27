# second architecture
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

# Load data
train_data = pd.read_csv('/content/mnist_train.csv')
test_data = pd.read_csv('/content/mnist_test.csv')

# Drop unnecessary columns
X = train_data.drop(columns=['label'])
y = train_data['label']
X_test = test_data.drop(columns=['label'])
y_test = test_data['label']

# Data Preprocessing
# Number of unique classes
num_classes = y.nunique()
print("Number of unique classes:", num_classes)

# Number of features
num_features = len(X) - 1  # Subtract 1 for the label column
print("Number of features:", num_features)

# Check for missing values
missing_values = train_data.isnull().sum().sum()
print("Number of missing values:", missing_values)

# Normalize pixel values
X_normalized = X / 255.0
X_test_normalized = X_test / 255.0

# Resize images to dimensions of 28 by 28
# Flatten feature data
X_flattened = X_normalized.values.reshape(-1, 28, 28)
X_test_flattened = X_test_normalized.values.reshape(-1, 28, 28)

# Visualize some resized images
plt.figure(figsize=(10, 4))
for i in range(1, 6):
    plt.subplot(1, 5, i)
    plt.imshow(X_flattened[i], cmap='gray')  # Corrected line
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
plt.show()

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# One-hot encode labels before splitting
y_train_encoded = to_categorical(y_train, num_classes=10)
y_val_encoded = to_categorical(y_val, num_classes=10)
y_test_encoded = to_categorical(y_test, num_classes=10)

# Define an ANN model with more hidden layers
model_ann = models.Sequential()
model_ann.add(layers.Dense(128, activation='relu', input_shape=(28 * 28,)))
model_ann.add(layers.Dense(64, activation='relu'))
model_ann.add(layers.Dense(32, activation='relu'))  # Additional layer
model_ann.add(layers.Dense(16, activation='relu'))  # Additional layer
model_ann.add(layers.Dense(10, activation='softmax'))

# Increase the learning rate
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile the ANN model with the custom optimizer
model_ann.compile(optimizer=custom_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Increase the batch size and train the ANN model
model_ann.fit(X_train, y_train_encoded, epochs=50, batch_size=128, validation_data=(X_val, y_val_encoded))

# Evaluate The ANN model on the validation set
y_val_pred_probs = model_ann.predict(X_val)
y_val_pred = np.argmax(y_val_pred_probs, axis=1)

# Confusion Matrix
conf_mat = confusion_matrix(y_val, y_val_pred)

# Display Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Evaluate The ANN model on the validation set
val_loss, val_acc = model_ann.evaluate(X_val, y_val_encoded)
print("Validation accuracy:", val_acc)

# Evaluate the ANN model on the test set
test_loss, test_acc = model_ann.evaluate(X_test_normalized, y_test_encoded)
print("Test accuracy:", test_acc)