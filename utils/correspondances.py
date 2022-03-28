import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_io as tfio
from sklearn.neighbors import KNeighborsClassifier
import sklearn
import sys


def generate_argmax_distribution(test_image, train_results_reducted, train_classes):
    min_image_dist = sys.float_info.max
    min_image_index = 0
    for i in range(len(train_results_reducted)):
        image_dist = 0.0
        for descriptor_source in test_image:
            min_eucl_dist = sys.float_info.max
            for descriptor_dest in train_results_reducted[i]:
                dist = np.linalg.norm(descriptor_source - descriptor_dest)
                min_eucl_dist = min(min_eucl_dist,dist)
            image_dist = image_dist + min_eucl_dist
        if image_dist<min_image_dist:
            min_image_dist=image_dist
            min_image_index=i

    return train_classes[min_image_index]

def evaluate_sequences(train_results_reducted, train_classes, test_results_reducted, test_classes):
    maximums = []
    for test_image in test_results_reducted:
        max = generate_argmax_distribution(test_image, train_results_reducted, train_classes)
        maximums.append(max)

    accuracy = sklearn.metrics.accuracy_score(test_classes, maximums)
    return accuracy