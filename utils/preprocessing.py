import os

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow.keras.backend as K

def load_data(data_path, img_shape, prefix):
    img_rows = img_shape[0]
    img_cols = img_shape[1]
    labels = []
    data = []

    img_list = os.listdir(data_path)
    for img in img_list:
        input_img = cv2.imread(data_path + '/' + img)
        input_img = cv2.resize(input_img, (img_rows, img_cols))
        input_img = tf.keras.applications.vgg16.preprocess_input(input_img)
        data.append(input_img)
        index = int(img.replace(prefix, '').replace('-', '').replace('.jpg', '').replace('.png', '').replace('.PNG', '').replace('.JPG', ''))
        labels.append(index)

    data = np.array(data)
    data = data.astype('float32')
    labels = np.array(labels)
    return data, labels


def extract_descriptors(X, y, k, conv_base):
    get_layer_output = K.function([conv_base.input], [conv_base.layers[5].output])

    train_results = []
    train_results_norms = []
    train_classes = []


    for i in range(len(X)):
        output = get_layer_output(tf.expand_dims(X[i], 0, name=None))[0]
        laplacian = tfio.experimental.filter.laplacian(output, 5)
        normes = tf.norm(laplacian, ord='euclidean', axis=3)
        train_results_norms.append(normes)
        train_results.append(output[0])
        train_classes.append(y[i])

    train_results = np.array(train_results)
    train_results_norms = np.array(train_results_norms)

    norms_shape = tf.shape(train_results_norms)

    train_results = tf.transpose(train_results, (0, 3, 1, 2))
    descriptors_shape = tf.shape(train_results)

    train_results = tf.reshape(train_results, (
        descriptors_shape[0], descriptors_shape[1], descriptors_shape[2] * descriptors_shape[3]))

    train_results_norms = tf.reshape(train_results_norms,
                                     (norms_shape[0], norms_shape[1], norms_shape[2] * norms_shape[3]))

    top_values, top_indices = tf.nn.top_k(train_results_norms, k)

    train_results_reducted = []
    shape_top = top_indices.shape

    for i in range(shape_top[0]):
        descriptors = []
        for j in range(shape_top[2]):
            chosen_descr = train_results[i, :, top_indices[i, 0, j]]
            norme_descr = tf.norm(chosen_descr, ord='euclidean')
            descriptors.append(chosen_descr/norme_descr)

        train_results_reducted.append(descriptors)

    return train_results_reducted, train_classes