from sklearn import datasets
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.random import seed
from tensorflow import set_random_seed
import random as rn
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

seed(1)
set_random_seed(2)
rn.seed(12345)
tf.set_random_seed(1234)

#import danych wejsciowych
pd.set_option('display.width', 400)
colnames = ["rarency", "frequency", "monetary", "time", "donated blood in March 2007"]
patients = pd.read_csv("transfusion.data", names=colnames)

#wyswietlanie wykresow danych wejsciowych
sb.pairplot(patients, diag_kind="kde")
plt.show()

#wyswietlanie opisu danych wejsciowych, ich sredniej, minimalnej i maksymalnej wartosci itd
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#    print(patients.describe())

#podzial na dane i etykiety
patientsData = patients.iloc[:, :4]
patientsTarget = patients.iloc[:, 4]

#skalowanie danych w celu poprawienia jakosci modelu
patientsData = StandardScaler().fit_transform(patientsData)

#podzial na zbior testowy i trenujacy
patients_train_data, patients_test_data, patients_train_target, patients_test_target = train_test_split(patientsData, patientsTarget, test_size=0.1)

#utworzenie modelu sieci neuronowej. podanie ilosci warstw, liczby neuronow w kazdej warstwie i ich funkcji aktywacji
#TODO: pierwsza i ostatnia warstwa maja pozostac tak jak sa. sprawdzic wyniki dla roznej kombinacji warstw pomiedzy, roznej ilosci neuronow i roznych funkcji aktywacji, roznych dropoutow.
#TODO: porobic screeny wykresow dla roznych funkcji aktywacji i napisac do nich jakiej kombinacji neuronow uzyles. mozesz to wkleic gdzies na dysk google potem i ja to przejrze
neural_model = Sequential([
    Dense(2, input_shape=(4,), activation="relu"),
    Dropout(0.1),
    Dense(2, activation="selu"),
    Dropout(0.1),
    Dense(1, activation="sigmoid")
])

#opis wykorzystywanej sieci neuronowej
neural_model.summary()

#kompilacja sieci
neural_model.compile(SGD(lr = .003), "binary_crossentropy", metrics=["accuracy"])

#proces uczenia sie sieci neuronowej
np.random.seed(0)
run_hist_1 = neural_model.fit(patients_train_data, patients_train_target, epochs=4000, validation_data=(patients_test_data, patients_test_target), verbose=True, shuffle=False)

#wyniki poprawnosci predykcji modelu
print('Accuracy over training data is ', accuracy_score(patients_train_target, neural_model.predict_classes(patients_train_data)))

print('Accuracy over testing data is ', accuracy_score(patients_test_target, neural_model.predict_classes(patients_test_data)))

#wykres przedstawiajacy proces uczenia
plt.plot(run_hist_1.history["loss"], 'r', marker='.', label="Train Loss")
plt.plot(run_hist_1.history["val_loss"], 'b', marker='.', label="Validation Loss")
plt.title("Neural network learning with SGD")
plt.legend()
plt.grid()
plt.show()
