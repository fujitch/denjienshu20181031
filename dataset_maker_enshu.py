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
num_of_charges =  # 電荷の個数
length_of_side =  # 立方体の一片の長さ(m)
max_coulomb =  # 電荷の最大値
pi =  # 円周率
epsilon =  # 真空誘電率
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

    ####################################
    
    """
    課題3:電場となる変数fieldsを定義し、(11, 11, 3, 6)のゼロ行列とする。
    11, 11:立方体の１面の座標
    3:電場ベクトルxyz成分
    6:立方体6面
    をそれぞれ表している。
    """
    ############## 課題3 ################

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

####################################