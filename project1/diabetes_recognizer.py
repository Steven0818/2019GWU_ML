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
import tensorflow as tf
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

file_name = "diabetes.csv"
data = pd.read_csv(file_name)
print("finish loading training data...")
print("data preprocessing ===============")
print("remove Insulin feature...")
data = data.drop(['Insulin'], axis=1)
X = np.array(data.ix[:,:-1])
y = np.array(data['Outcome'])

# ramdom_state = 80 is the best
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=80)


X_train_preg = X_train[:, 0]
X_train_others = X_train[:, 1:]
X_train_others[X_train_others == 0] = np.nan
print("input missing data with mean...")
imp = Imputer(missing_values=np.nan, strategy='mean')
imp.fit(X_train_others)
X_train_others = imp.transform(X_train_others)
print("standardize all data...")
X_train_preg = np.asarray([X_train_preg]).T
X_train = np.concatenate((X_train_preg, X_train_others), axis=1)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print("finish data preprocessing ========")

print("start training knn model... ")

init_time = time.time()
knn = KNeighborsClassifier(n_neighbors = 9)
knn.fit(X_train, y_train)
print("training_time " + str(time.time() - init_time))
init_time = time.time()
pred = knn.predict(X_test)
print("prediction_time " + str(time.time() - init_time))
print(y_test)
print(pred)
cnf_matrix = confusion_matrix(y_test, pred)
accuracy = accuracy_score(y_test, pred)
class_names = ["0", "1"]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
print(accuracy)
plt.show()





########## nn model ###########
# feature_number = X.shape[1]
# xs = tf.placeholder(tf.float32, [None, feature_number])
# W1 = tf.get_variable("W1", shape=[feature_number, 20],
#            initializer=tf.contrib.layers.xavier_initializer())
# b1 = tf.get_variable("b1", shape=[20],
#            initializer=tf.contrib.layers.xavier_initializer())
# y = tf.nn.tanh(tf.matmul(xs, W1) + b1)
# W2 = tf.get_variable("W2", shape=[20, 10],
#            initializer=tf.contrib.layers.xavier_initializer())
# b2 = tf.get_variable("b2", shape=[10],
#            initializer=tf.contrib.layers.xavier_initializer())
# y = tf.nn.tanh(tf.matmul(y, W2) + b2)
# ys = tf.placeholder(tf.float32, [None])
# W3 = tf.get_variable("W3", shape=[10, 1],
#            initializer=tf.contrib.layers.xavier_initializer())
# b3 = tf.get_variable("b3", shape=[1],
#            initializer=tf.contrib.layers.xavier_initializer())
# yi = tf.squeeze(tf.matmul(y, W3) + b3)
# cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits= yi, labels = ys))
# predict_result = tf.nn.sigmoid(yi)
# train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
# # 使用 with 可以讓Session自動關閉
# with tf.Session() as sess:

#     ################################
    
    # sess.run(tf.global_variables_initializer())
    # for j in range(10):
    #     for i in range(20):
            
    #         # 在 tensorflow內要使用run，才會讓計算圖開始執
    #         # print(X_train[i * 50:(i + 1) * 50,:].shape)
    #         # print(y_train.shape)
    #         print(i)
    #         # print(sess.run(W2))
    #         sess.run(train_step, feed_dict={xs: X_train[i * 30:(i + 1) * 30,:], ys: y_train[i * 30:(i + 1) * 30:]})
    # print(y_test)
    # predict_logit = sess.run(predict_result, feed_dict={xs: X_test, ys: y_test})
    # predict_logit[predict_logit >= 0.5] = 1
    # predict_logit[predict_logit < 0.5] = 0
    # cnf_matrix = confusion_matrix(y_test, predict_logit)
    # accuracy = accuracy_score(y_test, predict_logit)
    # print(accuracy)
    # plt.figure()
    # class_names = ["0", "1"]
    # plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                   title='Confusion matrix, without normalization')
    # plt.show()