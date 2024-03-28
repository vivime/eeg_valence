

"""
This file is an adaptation of  https://www.kaggle.com/code/shreyaspj/detecting-emotions-using-eeg-waves/notebook
"""


import tensorflow as tf
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Flatten, Dense
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow import keras
import seaborn as sns
from sklearn.preprocessing import LabelEncoder



data = pd.read_csv('/Users/vivienmezei/development/ba-vivien-mezei/features.csv')
print(data.info())

# #Encoding the 3 distinct labels
# #The 3 labels are : "NEGATIVE", "NEUTRAL" and "POSITIVE".
# le = LabelEncoder()
# data['label']=le.fit_transform(data['label'])


le = LabelEncoder()
data['Dominance']=le.fit_transform(data['Dominance'])

#Defining necessary features for model training
X = data.drop(['Valence', 'Arousal', 'Dominance'], axis=1)

# Target variable
y = data['Dominance']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=48)
X_train = np.array(X_train).reshape((X_train.shape[0],X_train.shape[1],1))
X_test = np.array(X_test).reshape((X_test.shape[0],X_test.shape[1],1))
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

#Defining the Model's architecture
inputs = tf.keras.Input(shape=(X_train.shape[1],1))

gru = tf.keras.layers.GRU(256, return_sequences=True)(inputs)
flat = Flatten()(gru)
outputs = Dense(3, activation='softmax')(flat)

model = tf.keras.Model(inputs, outputs)
model.summary()

#Training the model
def train_model(model, x_train, y_train, x_test, y_test, save_to, epoch=2):
    opt_adam = keras.optimizers.Adam(learning_rate=0.001)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(save_to + '_best_model_dominance.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.001 * np.exp(-epoch / 10.))

    model.compile(optimizer=opt_adam,
                  loss=['categorical_crossentropy'],
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=32,
                        epochs=epoch,
                        validation_data=(x_test, y_test),
                        callbacks=[es, mc, lr_schedule])


    return model, history

model,history = train_model(model, X_train, y_train,X_test, y_test, save_to= './', epoch = 40)

#Plotting the validation curves
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

#Evaluating the Model
model_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print("Test Accuracy: {:.3f}%".format(model_acc * 100))

y_pred = np.array(list(map(lambda x: np.argmax(x), model.predict(X_test))))

y_pred = np.array(list(map(lambda x: np.argmax(x), model.predict(X_test))))
y_test = y_test.idxmax(axis=1)

cm = confusion_matrix(y_test, y_pred)
clr = classification_report(y_test, y_pred)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:\n----------------------\n", clr)