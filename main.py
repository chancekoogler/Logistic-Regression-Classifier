import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from utils import load_dataset, model


(
    train_set_x_orig,
    train_set_y,
    test_set_x_orig,
    test_set_y,
    classes,
) = load_dataset()

reshapedTrainData = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
reshapedTestData = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_data = reshapedTrainData/255
test_data = reshapedTestData/255

d = model(train_data, train_set_y, test_data, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
