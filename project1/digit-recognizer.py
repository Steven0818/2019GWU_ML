import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
import numpy as np  
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
import time
# import tensorflow as tf

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

train_file_name = "digit-recognizer/train.csv"
train_data = pd.read_csv(train_file_name)
print("finish loading training data...")

X = np.array(train_data.ix[:, 1:])
y = np.array(train_data['label'])
print("start training knn model... ")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
init_time = time.time()
knn = KNeighborsClassifier(n_neighbors=13)
knn.fit(X_train, y_train)
print("training_time " + str(time.time() - init_time))
init_time = time.time()
pred = knn.predict(X_test)
print("prediction_time " + str(time.time() - init_time))
accuracy = accuracy_score(y_test, pred)



cnf_matrix = confusion_matrix(y_test, pred)
accuracy = accuracy_score(y_test, pred)
class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
print(accuracy)
plt.show()

# print("start loading testing data...")
# test_file_name = "digit-recognizer/test.csv"
# test_data = pd.read_csv(test_file_name)
# print(test_data)
# x_test = np.array(test_data)[0:1000]
# print("finish loading testing data...")
# pred = knn.predict(x_test)
# print("predicting result...")
# print(pred)


