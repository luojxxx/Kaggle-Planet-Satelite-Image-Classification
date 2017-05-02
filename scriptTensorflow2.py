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

    return rgb

def splitSet(dataset, split1, split2):
    idx_split1 = int( len(dataset) * split1)
    idx_split2 = int( len(dataset) * split2)

    training = dataset[0:idx_split1]
    validation = dataset[idx_split1:idx_split2]
    test = dataset[idx_split2:] 

    return [ np.array(training) , np.array(validation), np.array(test) ]

# Extract training and test set
print('Setup Dataset')
rerun = False

saveImgRawPath = folderpath+'pickleImgRaw'
if rerun == True or not os.path.isfile(saveImgRawPath):
    train_ImgRaw = get_raw(train, train_path)
    pickle.dump(train_ImgRaw, open( saveImgRawPath , 'wb'))
else:
    train_ImgRaw = pickle.load(open(saveImgRawPath, 'rb'))

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

# Splitting training set into training and validation set
train_dataset, valid_dataset, test_dataset = splitSet(train_ImgRaw, 0.6, 0.8)
train_labels, valid_labels, test_labels = splitSet(y, 0.6, 0.8)

image_size = 64
num_labels = 17
num_channels = 3 # rgb

def reformat(dataset, labels):
    dataset = dataset.reshape( (-1, image_size, image_size, num_channels)).astype(np.float32)
    # labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


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
    loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


# Running network
num_steps = 1001

def accuracy(predictions, labels):
    return 0.50

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
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
            print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
            pickle.dump(test_prediction.eval(), open('/Users/cloudlife/GitHub/kaggleplanet/predpk', 'wb'))


# # Making Final Predictions using all training data
# print('Extracting test features')
# test_features = get_Image(test, test_path)

# print('Training')
# clf = clf.fit(X, y)

# print('Predicting')
# X_predictions = np.array(test_features.drop(['image_name', 'tags'], axis=1))
# y_predictions = [ clf.predict(test_chip.reshape(1,-1)) for test_chip in tqdm(X_predictions) ]

# preds = [' '.join(labels[y_pred_row[0] > 0.2]) for y_pred_row in y_predictions]

# #Outputting predictions to csv
# subm = pd.DataFrame()
# subm['image_name'] = test_features.image_name.values
# subm['tags'] = preds
# subm.to_csv(folderpath+'submission.csv', index=False)

