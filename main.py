import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QMainWindow
from PyQt5.QtWidgets import QMessageBox
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *
import numpy as np
from sklearn.base import BaseEstimator
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from joblib import load
import pandas as pd
import os


class BoostAndKnnStack(BaseEstimator):
  def __init__(self, boosting_model_params: dict = None, knn_model_params: dict = None, verbose: bool = False):
    self.boosting_model_params = boosting_model_params
    self.knn_model_params = knn_model_params

    if self.boosting_model_params == None:
        self.boosting_model = lgb.LGBMClassifier()
    else:
      self.boosting_model = lgb.LGBMClassifier(**self.boosting_model_params)

    if self.knn_model_params == None:
        self.knn_model = KNeighborsClassifier()
    else:
        self.knn_model = KNeighborsClassifier(**self.knn_model_params)
    self.verbose = verbose

  def fit(self, X, y):
    self.boosting_model.fit(X, y)
    self.knn_model.fit(X, y)
    pass

  def predict(self, X):
    naive_output = (self.boosting_model.predict_proba(X) + self.knn_model.predict_proba(X)) / 2
    return np.argmax(naive_output, axis=1)

  def predict_proba(self, X):
    return (self.boosting_model.predict_proba(X) + self.knn_model.predict_proba(X)) / 2


complete = False
measurements = None
df = None
X = None
classifier = load('final_model.joblib')
prediction = None
proba = None


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("gui.ui", self)
        self.browse.clicked.connect(self.select_target)
        self.startButton.clicked.connect(self.predict)
        self.saveButton.clicked.connect(self.save_result)

    def predict(self):
        global prediction
        global proba
        try:
            prediction = classifier.predict(X)
            proba = classifier.predict_proba(X)
            proba = proba[:, 1]
            print(proba)
        except Exception as e:
            print(e)
            msg = QMessageBox()
            msg.setWindowTitle('Предупреждение')
            msg.setText("Ошибка: выберите входные данные для анализа.")
            xxx = msg.exec_()
        return 0

    def save_result(self):
        global prediction
        global df
        global proba
        if prediction is None:
            msg = QMessageBox()
            msg.setWindowTitle('Предупреждение')
            msg.setText("Ошибка: попытка сохранить результат до проведения анализа.")
            xxx = msg.exec_()
        else:
            width = df['X'].unique().shape[0]
            height = df['Y'].unique().shape[0]
            try:
                name = QFileDialog.getSaveFileName(self, 'Save File', r"C:\\", "TXT files (*.txt)")
                print("Path to save: ")
                print(name)
                if name != ('', '') and prediction is not None:
                    try:
                        output = np.reshape(prediction, (-1, width))
                        #pd.DataFrame(output).to_csv(name[0])
                        np.savetxt(name[0], output.astype(int), fmt='%d', delimiter=",")
                    except:
                        print('Invalid directory')
                        msg = QMessageBox()
                        msg.setWindowTitle('Предупреждение')
                        msg.setText("Ошибка: некорректный путь для сохранения результата.")
                        xxx = msg.exec_()
            except Exception as e:
                print(e)

    def select_target(self):
        global complete
        global measurements
        global df
        global X
        global prediction
        prediction = None
        df = None
        X = None
        try:
            complete = False
            data = QFileDialog.getOpenFileName(self, "Open file", r"C:\\", "TXT files (*.txt)")
            self.filename.setText(data[0])
            df = pd.read_csv(data[0], sep='\t', skiprows=[0],
                             header=None, names=['X', 'Y', 'Wave', 'Intensity'])
            measurements = []
            for i in range(df.shape[0] // 1015):
                measurements.append(
                    df[['Intensity']][i * len(df['Wave'].unique()):(i + 1) * len(df['Wave'].unique())].to_numpy())
            X = np.asarray(measurements)
            oldShape = X.shape
            X = X.reshape(oldShape[0], oldShape[1])
        except:
            msg = QMessageBox()
            msg.setWindowTitle('Предупреждение')
            msg.setText("Ошибка: неверный формат входных данных.")
            xxx = msg.exec_()


def show_window():
    app = QApplication(sys.argv)
    mainwindow = MainWindow()
    widget = QtWidgets.QStackedWidget()
    widget.addWidget(mainwindow)
    widget.setFixedWidth(733)
    widget.setFixedHeight(180)
    widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    show_window()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
