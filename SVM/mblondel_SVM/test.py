"""
@File: test.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2018, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Created at: 2018-06-06 20:13
@Desc: 
"""
from SVM.dataset import generate_concentric_circles
from SVM.mblondel_SVM.mblondel_SVM import SVM
from SVM.mblondel_SVM.mblondel_SVM import gaussian_kernel

import numpy as np

data = generate_concentric_circles()
svm = SVM(kernel=gaussian_kernel)
features = []
result = []
print(features)
for i in data[0]:
    features.append(i)
    result.append(-1.)
for i in data[1]:
    features.append(i)
    result.append(1.)

features = np.array(features)
result = np.array(result)
print(features, result)
svm.fit(features, result)
