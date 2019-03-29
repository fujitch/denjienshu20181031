# -*- coding: utf-8 -*-
# 必要なライブラリのインポート
import math
import numpy as np
import random
import pickle
"""
課題1:初期値を設定する。
"""
############## 課題1 ################
num_of_charges = 2 # 電荷の個数
length_of_side = 0.1 # 立方体の一片の長さ(m)
max_coulomb = 10 # 電荷の最大値
pi = math.pi # 円周率
epsilon = pow(8.85418782, -12) # 真空誘電率
####################################

# x, y, z座標のリストを作成
x_coordinates = []
for x in range(11):
    x_coordinates.append(x * length_of_side / 10)
y_coordinates = []
for y in range(11):
    y_coordinates.append(y * length_of_side / 10)
z_coordinates = []
for z in range(11):
    z_coordinates.append(z * length_of_side / 10)
        
# 電荷を配置し、電場を作成する関数を定義
def make_charge_field():
    """
    課題2:電荷となる変数chargesを定義し、(4, num_of_charges)のゼロ行列とする。
    0行目:0~max_coulombの範囲でランダムに電荷を挿入
    1行目:x_coordinatesの中で両端を除いた座標を挿入
    2行目:y_coordinatesの中で両端を除いた座標を挿入
    3行目:z_coordinatesの中で両端を除いた座標を挿入
    """
    ############## 課題2 ################
    charges = np.zeros((4, num_of_charges))
    for i in range(num_of_charges):
        charges[0, i] = random.random() * max_coulomb
        charges[1, i] = x_coordinates[random.randint(1, 9)]
        charges[2, i] = y_coordinates[random.randint(1, 9)]
        charges[3, i] = z_coordinates[random.randint(1, 9)]
    ####################################
    
    """
    課題3:電場となる変数fieldsを定義し、(11, 11, 3, 6)のゼロ行列とする。
    11, 11:立方体の１面の座標
    3:電場ベクトルxyz成分
    6:立方体6面
    をそれぞれ表している。
    """
    ############## 課題3 ################
    fields = np.zeros((11, 11, 3, 6))
    ####################################
    
    # x座標を0に固定し、立方体１面の電場をfieldsに挿入する。
    x = x_coordinates[0]
    for y_index in range(len(y_coordinates)):
        for z_index in range(len(z_coordinates)):
            for num in range(num_of_charges):
                charge = charges[:, num]
                y = y_coordinates[y_index]
                z = z_coordinates[z_index]
                r = np.array([x - charge[1], y - charge[2], z - charge[3]])
                fields[y_index, z_index, 0, 0] += (charge[0] * r[0]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
                fields[y_index, z_index, 1, 0] += (charge[0] * r[1]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
                fields[y_index, z_index, 2, 0] += (charge[0] * r[2]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
    """
    課題4:上記スクリプトを参考にx座標を0.1固定、y座標を0, 0.1固定、z座標を0, 0.1固定して残り5面の電場をfieldsに挿入する。
    自分で書きたい場合は上記を消して実装してください。
    """
    ############## 課題4 ################
    x = x_coordinates[10]
    for y_index in range(len(y_coordinates)):
        for z_index in range(len(z_coordinates)):
            for num in range(num_of_charges):
                charge = charges[:, num]
                y = y_coordinates[y_index]
                z = z_coordinates[z_index]
                r = np.array([x - charge[1], y - charge[2], z - charge[3]])
                fields[y_index, z_index, 0, 1] += (charge[0] * r[0]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
                fields[y_index, z_index, 1, 1] += (charge[0] * r[1]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
                fields[y_index, z_index, 2, 1] += (charge[0] * r[2]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
    y = y_coordinates[0]
    for z_index in range(len(z_coordinates)):
        for x_index in range(len(x_coordinates)):
            for num in range(num_of_charges):
                charge = charges[:, num]
                z = z_coordinates[z_index]
                x = x_coordinates[x_index]
                r = np.array([x - charge[1], y - charge[2], z - charge[3]])
                fields[z_index, x_index, 0, 2] += (charge[0] * r[0]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
                fields[z_index, x_index, 1, 2] += (charge[0] * r[1]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
                fields[z_index, x_index, 2, 2] += (charge[0] * r[2]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
    y = y_coordinates[10]
    for z_index in range(len(z_coordinates)):
        for x_index in range(len(x_coordinates)):
            for num in range(num_of_charges):
                charge = charges[:, num]
                z = z_coordinates[z_index]
                x = x_coordinates[x_index]
                r = np.array([x - charge[1], y - charge[2], z - charge[3]])
                fields[z_index, x_index, 0, 3] += (charge[0] * r[0]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
                fields[z_index, x_index, 1, 3] += (charge[0] * r[1]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
                fields[z_index, x_index, 2, 3] += (charge[0] * r[2]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
    z = z_coordinates[0]
    for x_index in range(len(x_coordinates)):
        for y_index in range(len(y_coordinates)):
            for num in range(num_of_charges):
                charge = charges[:, num]
                x = x_coordinates[x_index]
                y = y_coordinates[y_index]
                r = np.array([x - charge[1], y - charge[2], z - charge[3]])
                fields[x_index, y_index, 0, 4] += (charge[0] * r[0]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
                fields[x_index, y_index, 1, 4] += (charge[0] * r[1]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
                fields[x_index, y_index, 2, 4] += (charge[0] * r[2]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
    z = z_coordinates[10]
    for x_index in range(len(x_coordinates)):
        for y_index in range(len(y_coordinates)):
            for num in range(num_of_charges):
                charge = charges[:, num]
                x = x_coordinates[x_index]
                y = y_coordinates[y_index]
                r = np.array([x - charge[1], y - charge[2], z - charge[3]])
                fields[x_index, y_index, 0, 5] += (charge[0] * r[0]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
                fields[x_index, y_index, 1, 5] += (charge[0] * r[1]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
                fields[x_index, y_index, 2, 5] += (charge[0] * r[2]) / (4 * pi * epsilon * pow(np.linalg.norm(r), 3))
    ####################################
    return charges, fields

"""
課題5:ゼロ行列dataset_charges, dataset_fieldsを定義する。
それぞれ、(10000, 4, num_of_charges), (10000, 11, 11, 3, 6)のゼロ行列とする。
make_charge_field関数を10000回ループし、chargesとfieldsをそれぞれdataset_charges, dataset_fieldsに挿入していく。
dataset_charges, dataset_fieldsをtuple型としてdatasetという変数に挿入する。
dataset変数をpickleファイルとしてローカルストレージに保存する。ファイル名は任意。
"""
############## 課題5 ################
dataset_charges = np.zeros((10000, 4, num_of_charges))
dataset_fields = np.zeros((10000, 11, 11, 3, 6))
for i in range(100):
    charges, fields = make_charge_field()
    dataset_charges[i, :, :] = charges
    dataset_fields[i, :, :, :, :] = fields
dataset = (dataset_charges, dataset_fields)
pickle.dump(dataset, open("dataset20181031.pickle", "wb"))
####################################