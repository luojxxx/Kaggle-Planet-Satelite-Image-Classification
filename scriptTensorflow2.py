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
import cv2
from matplotlib import pyplot as plt

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

# Load data
folderpath = '/Users/cloudlife/GitHub/kaggleplanet/'
train_path = folderpath+'train-jpg/'
test_path = folderpath+'test-jpg/'
train = pd.read_csv(folderpath+'train.csv')
test = pd.read_csv(folderpath+'sample_submission.csv')

def get_raw(df, data_path):
    im_features = df.copy()

    rgb = []

    for image_name in tqdm(im_features.image_name.values, miniters=1000): 
        im = Image.open(data_path + image_name + '.jpg')
        im = im.resize((64,64))
        im = np.array(im)[:,:,:3]

        # im = np.hstack( ( im[:,:,0].ravel(), im[:,:,1].ravel(), im[:,:,2].ravel() ))
        rgb.append( im )

    return np.array(rgb)

def splitSet(dataset, split1, split2):
    idx_split1 = int( len(dataset) * split1)
    idx_split2 = int( len(dataset) * split2)

    training = dataset[0:idx_split1]
    validation = dataset[idx_split1:idx_split2]
    test = dataset[idx_split2:] 

    return [ training , validation, test ]

# Extract training and test set
print('Setup Dataset')
rerun = False

saveImgRawPath = folderpath+'pickleImgRaw'
saveSubmissionImgRawPath = folderpath+'pickleImgRawSubmission'
if rerun == True or not os.path.isfile(saveImgRawPath):
    train_ImgRaw = get_raw(train, train_path)
    submission_ImgRaw = get_raw(test, test_path)
    pickle.dump(train_ImgRaw, open( saveImgRawPath , 'wb'))
    pickle.dump(submission_ImgRaw, open( saveSubmissionImgRawPath , 'wb'))
else:
    train_ImgRaw = pickle.load(open(saveImgRawPath, 'rb'))
    submission_ImgRaw = pickle.load(open(saveSubmissionImgRawPath, 'rb'))

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

train_dataset, valid_dataset, test_dataset = splitSet(train_ImgRaw, 0.6, 0.8)
train_labels, valid_labels, test_labels = splitSet(y, 0.6, 0.8)

image_size = 64
num_labels = 17
num_channels = 3 # rgb

def reformat(dataset):
    dataset = dataset.reshape( (-1, image_size, image_size, num_channels)).astype(np.float32)
    # labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset

train_dataset, train_labels = [reformat(train_dataset), train_labels ]
valid_dataset, valid_labels = [reformat(valid_dataset), valid_labels ]
test_dataset, test_labels = [reformat(test_dataset), test_labels ]
submit_dataset = reformat(submission_ImgRaw)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
print('Submission set', submit_dataset.shape)


# Setting up network
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder( tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    # tf_submit_dataset = tf.constant(submit_dataset)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal( [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal( [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal( [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal( [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

    # Training computation.
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.sigmoid(logits)
    valid_prediction = tf.nn.sigmoid(model(tf_valid_dataset))
    test_prediction = tf.nn.sigmoid(model(tf_test_dataset))
    # submit_prediction = tf.nn.softmax(model(tf_submit_dataset))


# Running network
num_steps = 201

def accuracyMCMO(predictions, labels):
    count = 0
    for rowIdx, rowVal in enumerate(labels):
        for eleIdx, eleVal in enumerate(rowVal):
            if labels[rowIdx][eleIdx] == predictions[rowIdx][eleIdx]:
                count+=1
    total = len(labels) * len(labels[0])
    return count/total
def accuracy(predictions, labels):
    formatPredictions = []
    for row in predictions:
        tempRow = []
        for ele in row:
            if ele > 0.9:
                tempRow.append(1)
            else:
                tempRow.append(0)
        formatPredictions.append(tempRow)

    formatLabels = []
    for row in labels:
        tempRow = []
        for ele in row:
            if ele > 0.9:
                tempRow.append(1)
            else:
                tempRow.append(0)
        formatLabels.append(tempRow)

    # print('dump')
    # pickle.dump(formatPredictions, open( folderpath+'pred' , 'wb'))
    # pickle.dump(formatLabels, open( folderpath+'labels' , 'wb'))
    # print('done')

    return accuracyMCMO(formatPredictions, formatLabels) * 100


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run( [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.3f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.3f%%' % accuracy(valid_prediction.eval(), valid_labels))
        print('Test accuracy: %.3f%%' % accuracy(test_prediction.eval(), test_labels))

    saver = tf.train.Saver()
    saver.save(session, folderpath+'my-model')

    # submission_results = submit_prediction.eval()
    # pickle.dump(submission_results, open(folderpath+'submission', 'wb'))

# with tf.Session(graph=graph) as session:
#     saver = tf.train.Saver()
#     saver.restore(session, folderpath+'my-model')


# Making Final Predictions using all training data
print('Outputting Predictions')
y_predictions = pickle.load(open(folderpath+'submission', 'rb'))
preds = [' '.join( [labels[idx] for idx, val in enumerate(y_pred_row) if val > 0.9] ) for y_pred_row in y_predictions]

# Outputting predictions to csv
subm = pd.DataFrame()
subm['image_name'] = test.image_name.values
subm['tags'] = preds
subm.to_csv(folderpath+'submission.csv', index=False)


# ['selective_logging', 'conventional_mine', 'partly_cloudy',
#        'artisinal_mine', 'haze', 'slash_burn', 'primary', 'clear',
#        'bare_ground', 'blooming', 'water', 'road', 'cloudy', 'habitation',
#        'agriculture', 'blow_down', 'cultivation']