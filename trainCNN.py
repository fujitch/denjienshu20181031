# -*- coding: utf-8 -*-

# 必要ライブラリをインポート
import tensorflow as tf
import numpy as np
import pickle

# 初期値を設定する。
batch_size = 100
training_epochs = 3
display_epochs = 1
num_of_charges = 2

# pickleファイルを読み込む。
dataset_charges, dataset_fields = pickle.load(open('dataset20181031.pickle', 'rb'))

"""
課題6:dataset_charges, dataset_fieldsの規格化(0~1の値にする)を行う。
dataset_charges[:, 0, :]を10で割る。
dataset_charges[:, 1:, :]に10をかける。
dataset_fieldsの要素全てを最小値で引き、その後最大値で割る。
"""
############## 課題6 ################
dataset_charges[:, 0, :] /= 10
dataset_charges[:, 1:, :] *= 10
dataset_fields -= np.min(dataset_fields)
dataset_fields /= np.max(dataset_fields)
#####################################

"""
課題7:dataset_chargesを電荷の大きさでソートする。
dataset_charges.shape[0]回数ループを行い、電荷の大きさでソートし挿入し直す。
"""
############## 課題7 ################
for i in range(dataset_charges.shape[0]):
    charges = dataset_charges[i, :, :]
    dummyCharges = np.zeros((dataset_charges.shape[1], dataset_charges.shape[2]))
    indexes = np.argsort(charges[0, :])
    for k in range(len(indexes)):
        dummyCharges[:, k] = charges[:, indexes[k]]
    dataset_charges[i, :, :] = dummyCharges
#####################################
    
"""
課題8:dataset_charges, dataset_fieldsをそれぞれfloat型に変換する。
その後(10000, num_of_charges * 4), (10000, 11 * 11 * 3 * 6)にreshapeする。
"""
############## 課題8 ################
dataset_charges = np.array(dataset_charges, dtype=np.float32)
dataset_fields = np.array(dataset_fields, dtype=np.float32)
dataset_charges = np.reshape(dataset_charges, (10000, num_of_charges * 4))
dataset_fields = np.reshape(dataset_fields, (10000, 11 * 11 * 3 * 6))
#####################################

"""
課題9:データセットのうち9000を学習用、1000をテストデータとする。
学習用データをcharges_train, fields_train
テストデータをcharges_test, fields_test
という変数名とする。
"""
############## 課題9 ################
charges_train = dataset_charges[:9000, :]
charges_test = dataset_charges[9000:, :]
fields_train = dataset_fields[:9000, :]
fields_test = dataset_fields[9000:, :]
#####################################

tf.reset_default_graph()

# graphの構築
# placeholderの定義
x = tf.placeholder("float", shape=[None, 11 * 11 * 3 * 6])
y_ = tf.placeholder("float", shape=[None, 8])

# 荷重作成
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# バイアス作成
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 畳み込み処理を定義
def conv2d_pad(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# プーリング処理を定義
def max_pool_2_2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

"""
課題10:tensorflowグラフを構築する。
以下の実装例でも動作しますが、
時間があれば自身でフィルタの大きさ、特徴マップの大きさ、隠れ層の大きさ、畳み込み層を増やすなど工夫してください。
"""
############## 課題10 ################
# 畳み込み層1
W_conv1 = weight_variable([3, 3, 18, 256])
b_conv1 = bias_variable([256])
x_image = tf.reshape(x, [-1, 11, 11, 18])
h_conv1 = tf.nn.relu(conv2d_pad(x_image, W_conv1) + b_conv1)
# プーリング層1
h_pool1 = max_pool_2_2(h_conv1)

# 畳み込み層2
W_conv2 = weight_variable([3, 3, 256, 256])
b_conv2 = bias_variable([256])
h_conv2 = tf.nn.relu(conv2d_pad(h_pool1, W_conv2) + b_conv2)
# プーリング層2
h_pool2 = max_pool_2_2(h_conv2)

# 全結合層1
W_fc1 = weight_variable([256*3*3, 1024])
b_fc1 = bias_variable([1024])
h_flat = tf.reshape(h_pool2, [-1, 256*3*3])
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

# ドロップアウト
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 全結合層2
W_fc2 = weight_variable([1024, 8])
b_fc2 = bias_variable([8])
y_out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 学習誤差を求める
loss = tf.reduce_mean(tf.square(y_ - y_out))

# 最適化処理
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
#####################################

# セッション開始
with tf.Session() as sess:
    # グラフの保存クラス
    saver = tf.train.Saver()
    # グラフの初期化
    # sess.run(tf.initialize_all_variables())
    # 学習済みグラフの読み込み
    saver.restore(sess, "./20181031model")
    
    # training_epoch回学習を行う
    for i in range(training_epochs):
        """
        課題11:fields_train, charges_trainをbatch_sizeずつ取り出すし、batch, outputとする。
        batch:(batch_size, 1800)
        output:(batch_size, 8)
        となる。
        placeholderのx, y_をそれぞれbatch, outputとしてグラフのtrain_stepを呼び出すことで学習を行う。
        """
        for k in range(0, charges_train.shape[0], batch_size):
            batch = fields_train[k:k+batch_size, :]
            output = charges_train[k:k+batch_size, :]
            train_step.run(session=sess, feed_dict={x: batch, y_: output, keep_prob: 0.5})
        # display_epochsごとに学習経過を出力する。
        if i%display_epochs == 0:
            train_loss = 0.0
            test_loss = 0.0
            for k in range(0, charges_train.shape[0], batch_size):
                batch = fields_train[k:k+batch_size, :]
                output = charges_train[k:k+batch_size, :]
                train_loss += loss.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
            for k in range(0, charges_test.shape[0], batch_size):
                batch = fields_test[k:k+batch_size, :]
                output = charges_test[k:k+batch_size, :]
                test_loss += loss.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
            train_loss /= 90
            test_loss /= 10
            print("training_finished:" + str(i) + "epochs")
            print("train_loss=" + str(train_loss))
            print("test_loss=" + str(test_loss))
    # 学習したモデルを保存する。
    saver.save(sess, "./20181031model")