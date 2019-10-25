from keras import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.feature_extraction.tests import test_image

(train_images,train_labels),(test_images, test_labels) = mnist.load_data()
#display the first image in the training data
plt.imshow(test_images[0,:,:],cmap='gray')
plt.title('Ground Truth : {}'.format(test_labels[0]))
plt.show()