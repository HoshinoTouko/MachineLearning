"""
@File: load_mnist.py
@Author: HoshinoTouko
@License: (C) Copyright 2014 - 2018, HoshinoTouko
@Contact: i@insky.jp
@Website: https://touko.moe/
@Created at: 2018-06-13 14:30
@Desc: 
"""
import numpy as np
import struct
import os


def load_datasets():
    train_features = os.path.abspath(os.path.dirname(__file__)) + \
                     '/train-images.idx3-ubyte'
    train_label = os.path.abspath(os.path.dirname(__file__)) + \
                  '/train-labels.idx1-ubyte'
    test_features = os.path.abspath(os.path.dirname(__file__)) + \
                    '/t10k-images.idx3-ubyte'
    test_label = os.path.abspath(os.path.dirname(__file__)) + \
                 '/t10k-labels.idx1-ubyte'
    return load_image_set(train_features), \
           load_label_set(train_label), \
           load_image_set(test_features), \
           load_label_set(test_label)



def load_image_set(filename):
    print("load image set", filename)

    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0)
    print("head,", head)

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    # [60000]*28*28
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'  # like '>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset)

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, 1, width * height])
    print("load imgs finished")

    return imgs


def load_label_set(filename):
    print("load label set", filename)

    binfile = open(filename, 'rb')
    buffers = binfile.read()

    head = struct.unpack_from('>II', buffers, 0)
    print("head,", head)

    imgNum = head[1]

    offset = struct.calcsize('>II')
    numString = '>' + str(imgNum) + "B"
    labels = struct.unpack_from(numString, buffers, offset)
    binfile.close()
    labels = np.reshape(labels, [imgNum, 1])

    print('load label finished')

    return labels


if __name__ == "__main__":
    # imgs = load_image_set("train-images.idx3-ubyte")
    # labels = load_image_set("train-labels.idx1-ubyte")
    print(load_datasets())

    # imgs = loadImageSet("t10k-images.idx3-ubyte")
    # labels = loadLabelSet("t10k-labels.idx1-ubyte")
