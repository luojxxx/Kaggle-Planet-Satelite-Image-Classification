import numpy as np
import os
import pandas as pd
import random
import pickle
from tqdm import tqdm

import tensorflow as tf

import scipy
from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

from PIL import Image
# import cv2
from matplotlib import pyplot as plt

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

# Load data
folderpath = 'C:\\Users\\Romtean\\Documents\\GitHub\\kaggleplanet\\'
train_path = 'E:\\Kaggle Projects/Amazon 2017/images/multi-label/'
# test_path = 'E:\\Kaggle Projects/Amazon 2017/test-jpg/'
train = pd.read_csv(folderpath + 'train.csv')


# test = pd.read_csv(folderpath+'sample_submission.csv')

def get_raw(df, data_path):
    im_features = df.copy()

    rgb = []
    for image_name in tqdm(im_features.image_name.values, miniters=1000):
        im = Image.open(data_path + image_name + '.jpg')
        im = im.resize((64, 64))
        im = np.array(im)[:, :, :3]
        # im = np.hstack( ( im[:,:,0].ravel(), im[:,:,1].ravel(), im[:,:,2].ravel() ))
        rgb.append(im)

    return np.array(rgb)


def splitSet(dataset, split1, split2):
    idx_split1 = int(len(dataset) * split1)
    idx_split2 = int(len(dataset) * split2)

    training = dataset[0:idx_split1]
    validation = dataset[idx_split1:idx_split2]
    test = dataset[idx_split2:]

    return [training, validation, test]


# Extract training and test set
print('Setup Dataset')
rerun = False

saveImgRawPath = folderpath + 'pickleImgRaw'
saveSubmissionImgRawPath = folderpath + 'pickleImgRawSubmission'
if rerun == True or not os.path.isfile(saveImgRawPath):
    train_ImgRaw = get_raw(train, train_path)
    # submission_ImgRaw = get_raw(test, test_path)
    pickle.dump(train_ImgRaw, open(saveImgRawPath, 'wb'), protocol=4)
    # pickle.dump(submission_ImgRaw, open( saveSubmissionImgRawPath , 'wb'), protocol=4)
else:
    train_ImgRaw = pickle.load(open(saveImgRawPath, 'rb'))
    # submission_ImgRaw = pickle.load(open(saveSubmissionImgRawPath, 'rb'))

print('Setup Dataset Labels')
y_train = []

flatten = lambda l: [item for sublist in l for item in sublist]
labels = np.array(list(set(flatten([l.split(' ') for l in train['tags'].values]))))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for tags in tqdm(train.tags.values, miniters=1000):
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    y_train.append(targets)

y = np.array(y_train, np.float32)
# y = y[:,2]
# y = np.array( [ np.array([i]) for i in y ])

# Splitting training set into training and validation set
# ~~ test data ~~
# from sklearn.datasets import load_digits
# data = load_digits()
# train_ImgRaw = data.data
# y = []
# for label in data.target:
#     arr = np.zeros(10)
#     arr[label] = 1
#     y.append(arr)
# y = np.array(y)
# ~~ test data ~~

train_dataset, valid_dataset, test_dataset = splitSet(train_ImgRaw, 0.8, 0.9)
train_labels, valid_labels, test_labels = splitSet(y, 0.8, 0.9)

image_size = 64
num_labels = 17
num_channels = 3  # rgb
confidence_cutoff = 0.5  # what confidence to consider prediction as part of class


def reformat(dataset):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    # labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset


train_dataset, train_labels = [reformat(train_dataset), train_labels]
valid_dataset, valid_labels = [reformat(valid_dataset), valid_labels]
test_dataset, test_labels = [reformat(test_dataset), test_labels]
# submit_dataset = reformat(submission_ImgRaw)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
# print('Submission set', submit_dataset.shape)


# Setting up network
batch_size = 16
patch_size = 5
depth = 32
num_hidden = 128

graph = tf.Graph()


def output_size_pool(input_size, conv_filter_size, pool_filter_size, padding, conv_stride, pool_stride):
    if padding == 'same':
        padding = -1.00
    elif padding == 'valid':
        padding = 0.00
    else:
        return None
    # After convolution 1
    output_1 = (((input_size - conv_filter_size - 2 * padding) / conv_stride) + 1.00)
    # After pool 1
    output_2 = (((output_1 - pool_filter_size - 2 * padding) / pool_stride) + 1.00)
    # After convolution 2
    output_3 = (((output_2 - conv_filter_size - 2 * padding) / conv_stride) + 1.00)
    # After pool 2
    output_4 = (((output_3 - pool_filter_size - 2 * padding) / pool_stride) + 1.00)
    return int(output_4)


final_image_size = output_size_pool(input_size=image_size, conv_filter_size=5, pool_filter_size=2, padding='valid',
                                    conv_stride=1, pool_stride=2)
print(final_image_size)

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    '''Variables'''
    # Convolution 1 Layer
    # Input channels: num_channels = 1
    # Output channels: depth = 16
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))

    # Convolution 2 Layer
    # Input channels: depth = 16
    # Output channels: depth = 16
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

    # First Fully Connected Layer (Densely Connected Layer)
    # Use neurons to allow processing of entire image
    final_image_size = output_size_pool(input_size=image_size, conv_filter_size=5, pool_filter_size=2, padding='valid',
                                        conv_stride=1, pool_stride=2)
    layer3_weights = tf.Variable(
        tf.truncated_normal([final_image_size * final_image_size * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    # Second Fully Connected Layer
    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    # Readout layer: Sigmoid Layer
    layer5_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


    def model(data):
        # First Convolutional Layer with Pooling
        conv_1 = tf.nn.conv2d(data, layer1_weights, strides=[1, 1, 1, 1], padding='VALID')
        hidden_1 = tf.nn.relu(conv_1 + layer1_biases)
        pool_1 = tf.nn.avg_pool(hidden_1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

        # Second Convolutional Layer with Pooling
        conv_2 = tf.nn.conv2d(pool_1, layer2_weights, strides=[1, 1, 1, 1], padding='VALID')
        hidden_2 = tf.nn.relu(conv_2 + layer2_biases)
        pool_2 = tf.nn.avg_pool(hidden_2, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

        # First Fully Connected Layer
        shape = pool_2.get_shape().as_list()
        reshape = tf.reshape(pool_2, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        keep_prob = 0.5
        hidden_drop = tf.nn.dropout(hidden, keep_prob)

        # Second Fully Connected Layer
        hidden_2 = tf.nn.relu(tf.matmul(hidden_drop, layer4_weights) + layer4_biases)
        hidden_2_drop = tf.nn.dropout(hidden_2, keep_prob)

        # Readout Layer: Softmax Layer
        return tf.matmul(hidden_2_drop, layer5_weights) + layer5_biases


    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    global_step = tf.Variable(0)  # count the number of steps taken.
    start_learning_rate = 0.05
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 10000, 0.96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.sigmoid(logits)
    valid_prediction = tf.nn.sigmoid(model(tf_valid_dataset))
    test_prediction = tf.nn.sigmoid(model(tf_test_dataset))

# Running network
num_steps = 300


def accuracyMCMO(predictions, labels):
    count = 0
    total = 0
    for rowIdx, rowVal in enumerate(labels):
        for eleIdx, eleVal in enumerate(rowVal):
            if labels[rowIdx][eleIdx] == 1:
                total += 1

                if labels[rowIdx][eleIdx] == predictions[rowIdx][eleIdx]:
                    count += 1

    return count / total


def accuracy(predictions, labels):
    formatPredictions = []
    for row in predictions:
        tempRow = []
        for ele in row:
            if ele > confidence_cutoff:
                tempRow.append(1)
            else:
                tempRow.append(0)
        formatPredictions.append(tempRow)

    formatLabels = []
    for row in labels:
        tempRow = []
        for ele in row:
            if ele > confidence_cutoff:
                tempRow.append(1)
            else:
                tempRow.append(0)
        formatLabels.append(tempRow)

    # print('dump')
    # pickle.dump(formatPredictions, open( folderpath+'pred' , 'wb'))
    # pickle.dump(formatLabels, open( folderpath+'labels' , 'wb'))
    # print('done')
    # print(formatPredictions)
    # print('')
    # print(formatLabels)

    return accuracyMCMO(formatPredictions, formatLabels) * 100


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
with tf.Session(graph=graph, config=config) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.3f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.3f%%' % accuracy(valid_prediction.eval(), valid_labels))

    test_results = test_prediction.eval()
    print('Test accuracy: %.3f%%' % accuracy(test_results, test_labels))
    pickle.dump(test_results, open(folderpath + 'test_results', 'wb'), protocol=4)

    saver = tf.train.Saver()
    saver.save(session, folderpath + 'my-model')


# # Running Model on Submission Set
# print('Evaluating Submission Set')
# with tf.Session(graph=graph) as session:
#     saver = tf.train.Saver()
#     saver.restore(session, folderpath+'my-model')

#     tf_submit_dataset = tf.constant(submit_dataset)
#     submit_prediction = tf.nn.sigmoid(model(tf_submit_dataset))
#     submission = submit_prediction.eval()
#     pickle.dump(submission, open(folderpath+'submission', 'wb'))


# # Outputting Predictions to Csv
# print('Outputting Predictions')
# y_predictions = pickle.load(open(folderpath+'submission', 'rb'))
# preds = [' '.join( [labels[idx] for idx, val in enumerate(y_pred_row) if val > confidence_cutoff] ) for y_pred_row in y_predictions]

# subm = pd.DataFrame()
# subm['image_name'] = test.image_name.values
# subm['tags'] = preds
# subm.to_csv(folderpath+'submission.csv', index=False)


# Labels
# ['selective_logging', 'conventional_mine', 'partly_cloudy',
#        'artisinal_mine', 'haze', 'slash_burn', 'primary', 'clear',
#        'bare_ground', 'blooming', 'water', 'road', 'cloudy', 'habitation',
#        'agriculture', 'blow_down', 'cultivation']


# Analytics
def getLabelDistribution(labels, labelNameArray=None):
    labelCount = [np.sum(labels[:, i]) for i in range(0, len(labels[0]))]
    if labelNameArray == None:
        return labelCount
    else:
        labelNameCount = {key: val for key, val in zip(labelNameArray, labelCount)}
        return labelCount, labelNameCount

        # Training set label distribution
        # {'slash_burn': 209.0, 'blooming': 332.0, 'water': 7262.0, 'cloudy': 2330.0, 'selective_logging': 340.0,
        #  'road': 8076.0, 'primary': 37840.0, 'clear': 28203.0, 'haze': 2695.0, 'agriculture': 12338.0, 'cultivation': 4477.0,
        #  'partly_cloudy': 7251.0, 'bare_ground': 859.0, 'conventional_mine': 100.0, 'artisinal_mine': 339.0,
        #  'habitation': 3662.0, 'blow_down': 98.0}

        # Potential additions edge and line analysis can be combined with RGB statistics.
        # Canny edge analysis and count how many 1s are there.
        # Line edge analysis and count how many 1s are there.
        # Corner analysis and count how many 1s are there.