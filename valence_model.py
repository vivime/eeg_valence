
"""
This file is an adaptation of  https://www.kaggle.com/code/shreyaspj/detecting-emotions-using-eeg-waves/notebook
"""
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Flatten, Dense
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('features.csv')
print(data.info())

# Encoding the 3 distinct labels
# #The 3 labels are : "NEGATIVE", "NEUTRAL" and "POSITIVE".
le = LabelEncoder()
data['Valence'] = le.fit_transform(data['Valence'])

# Defining necessary features for model training
X = data.drop(['Valence', 'Arousal', 'Dominance'], axis=1)

# Target variable
y = data['Valence']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=48)


# SVM
svm_classifier = svm.SVC(kernel='poly', degree=3)
svm_classifier.fit(X_train, y_train)
joblib.dump(svm_classifier, 'models/svm_model_poly.pkl')

svm_predictions = svm_classifier.predict(X_test)
svm_accuracy = np.mean(svm_predictions == y_test)
print("SVM poly Test Accuracy: {:.3f}%".format(svm_accuracy * 100))
print(classification_report(y_test, svm_predictions))

svm_cm = confusion_matrix(y_test, svm_predictions)
svm_clr = classification_report(y_test, svm_predictions)
# Plot confusion matrix for SVM
plt.figure(figsize=(8, 8))
sns.heatmap(svm_cm, annot=True, fmt='g', cbar=False, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM poly Confusion Matrix")
plt.show()

# Print classification report for SVM
print("SVM rbf Classification Report:\n----------------------\n", svm_clr)

X_train = np.array(X_train).reshape((X_train.shape[0] ,X_train.shape[1] ,1))
X_test = np.array(X_test).reshape((X_test.shape[0] ,X_test.shape[1] ,1))
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# GRU
# Defining the Model's architecture
inputs = tf.keras.Input(shape=(X_train.shape[1] ,1))

gru = tf.keras.layers.GRU(256, return_sequences=True)(inputs)
flat = Flatten()(gru)
outputs = Dense(3, activation='softmax')(flat)

model = tf.keras.Model(inputs, outputs)
model.summary()


# Training the model
def train_model(model, x_train, y_train, x_test, y_test, save_to, epoch=2):
    opt_adam = keras.optimizers.Adam(learning_rate=0.001)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(save_to + 'gru_model_valence.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * np.exp(-epoch / 10.))

    model.compile(optimizer=opt_adam,
                  loss=['categorical_crossentropy'],
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=32,
                        epochs=epoch,
                        validation_data=(x_test, y_test),
                        callbacks=[es, mc, lr_schedule] ,)

    return model, history


model ,history = train_model(model, X_train, y_train ,X_test, y_test, save_to= './models/', epoch = 40)

# Plotting the validation curves
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Evaluating the Model
model_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print("Test Accuracy: {:.3f}%".format(model_acc * 100))

y_pred = np.array(list(map(lambda x: np.argmax(x), model.predict(X_test))))

y_test = y_test.idxmax(axis=1)

cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix GRU")
plt.show()

print("Classification Report GRU:\n----------------------\n", clr)


# Define the CNN model
model_CNN = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model_CNN.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display model summary
model_CNN.summary()

# Train the model
history = model_CNN.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
model_CNN.save('models/cnn_model_valence.h5')


# Evaluate the model
test_loss, test_acc = model_CNN.evaluate(X_test, y_test)
print('\nTest accuracy CNN:', test_acc)

# Plotting the validation curves
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('CNN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('CNN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Generate predictions
y_pred = np.argmax(model_CNN.predict(X_test), axis=1)
y_test = np.argmax(np.array(y_test), axis=1)

# Generate confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='g', cbar=False, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix CNN")
plt.show()

# Print classification report
print("Classification Report CNN:\n----------------------\n", clr)