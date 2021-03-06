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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier

from PIL import Image
import cv2
from matplotlib import pyplot as plt

# Potential additions edge and line analysis can be combined with RGB statistics.
# Canny edge analysis and count how many 1s are there.
# Line edge analysis and count how many 1s are there.
# Corner analysis and count how many 1s are there.
# Modify RGB statistics to Purple, Blue, Green, Yellow, Red, Brown, White, Black?
# Check misclassification statistics
# Utilize an ensemble algorithm, so maybe a Random forest for color + edge statistics, and a separate
# like a CNN trained specifically to look for specific labels like blow down. This image feature algorithm
# may potentially use artificially generated data.

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
    for image_name in tqdm(im_features.image_name.values, mininterval=10): 
        im = Image.open(data_path + image_name + '.jpg')
        im = im.resize((64,64))
        im = np.array(im)[:,:,:3]
        # im = np.hstack( ( im[:,:,0].ravel(), im[:,:,1].ravel(), im[:,:,2].ravel() ))
        rgb.append( im )

    return np.array(rgb)

def getEdges(df, data_path):
    im_features = df.copy()

    edgeCountArr = []

    for image_name in tqdm(im_features.image_name.values, mininterval=10): 
        img = cv2.imread( data_path + image_name + '.jpg' , 0)
        edges = cv2.Canny( img, 5, 25)

        edges_small = cv2.resize(edges, (0,0), fx=0.25, fy=0.25)
        edgeIndices = edges_small > 125
        edges_small[edgeIndices] = 1

        edgeCountArr.append( np.array([np.sum(edges_small)]) )
    
    return np.array(edgeCountArr)

def getDistance(xypair):
    x_delta = abs(xypair[0] - xypair[2])
    y_delta = abs(xypair[1] - xypair[3])
    hypotenuse = (x_delta**2 + y_delta**2)**0.5
    return hypotenuse

def getLines(df, data_path):
    im_features = df.copy()

    lineDistanceArr = []
    for image_name in tqdm(im_features.image_name.values, mininterval=10): 
        img = cv2.imread( data_path + image_name + '.jpg' , 0)

        edges = cv2.Canny( img, 5, 25)
        edges_small = cv2.resize(edges, (0,0), fx=0.25, fy=0.25)

        lines = cv2.HoughLinesP(edges_small,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
        if lines is None:
            lineDistanceArr.append(  np.array([0]) )
        else:
            lineDistanceSum = np.sum( [ getDistance(line[0]) for line in lines ] )
            lineDistanceArr.append(  np.array([lineDistanceSum]) )
    
    return np.array(lineDistanceArr)

def getCorners(df, data_path):
    im_features = df.copy()

    cornerSumArr = []

    for image_name in tqdm(im_features.image_name.values, mininterval=10): 
        img = cv2.imread( data_path + image_name + '.jpg' , 0)
        img = np.float32(img)
        dst = cv2.cornerHarris(img,2,3,0.04)

        thresholdIndices = dst > 0.05 * dst.max()
        matrix = np.zeros(shape=(dst.shape[0],dst.shape[1]))
        matrix[thresholdIndices] = 1
        cornerSum = np.sum(matrix)
        cornerSumArr.append( np.array([cornerSum]) )

    return np.array(cornerSumArr)

def extract_features(df, data_path):
    im_features = df.copy()

    r_mean = []
    g_mean = []
    b_mean = []

    r_std = []
    g_std = []
    b_std = []

    r_max = []
    g_max = []
    b_max = []

    r_min = []
    g_min = []
    b_min = []

    r_kurtosis = []
    g_kurtosis = []
    b_kurtosis = []
    
    r_skewness = []
    g_skewness = []
    b_skewness = []

    for image_name in tqdm(im_features.image_name.values, mininterval=10): 
        im = Image.open(data_path + image_name + '.jpg')
        im = np.array(im)[:,:,:3]

        r_mean.append(np.mean(im[:,:,0].ravel()))
        g_mean.append(np.mean(im[:,:,1].ravel()))
        b_mean.append(np.mean(im[:,:,2].ravel()))

        r_std.append(np.std(im[:,:,0].ravel()))
        g_std.append(np.std(im[:,:,1].ravel()))
        b_std.append(np.std(im[:,:,2].ravel()))

        r_max.append(np.max(im[:,:,0].ravel()))
        g_max.append(np.max(im[:,:,1].ravel()))
        b_max.append(np.max(im[:,:,2].ravel()))

        r_min.append(np.min(im[:,:,0].ravel()))
        g_min.append(np.min(im[:,:,1].ravel()))
        b_min.append(np.min(im[:,:,2].ravel()))

        r_kurtosis.append(scipy.stats.kurtosis(im[:,:,0].ravel()))
        g_kurtosis.append(scipy.stats.kurtosis(im[:,:,1].ravel()))
        b_kurtosis.append(scipy.stats.kurtosis(im[:,:,2].ravel()))
        
        r_skewness.append(scipy.stats.skew(im[:,:,0].ravel()))
        g_skewness.append(scipy.stats.skew(im[:,:,1].ravel()))
        b_skewness.append(scipy.stats.skew(im[:,:,2].ravel()))


    im_features['r_mean'] = r_mean
    im_features['g_mean'] = g_mean
    im_features['b_mean'] = b_mean

    im_features['r_std'] = r_std
    im_features['g_std'] = g_std
    im_features['b_std'] = b_std

    im_features['r_max'] = r_max
    im_features['g_max'] = g_max
    im_features['b_max'] = b_max

    im_features['r_min'] = r_min
    im_features['g_min'] = g_min
    im_features['b_min'] = b_min

    im_features['r_kurtosis'] = r_kurtosis
    im_features['g_kurtosis'] = g_kurtosis
    im_features['b_kurtosis'] = b_kurtosis
    
    im_features['r_skewness'] = r_skewness
    im_features['g_skewness'] = g_skewness
    im_features['b_skewness'] = b_skewness
    
    return np.array(im_features.drop(['image_name', 'tags'], axis=1))

def splitSet(dataset, split1, split2):
    idx_split1 = int( len(dataset) * split1)
    idx_split2 = int( len(dataset) * split2)

    training = dataset[0:idx_split1]
    validation = dataset[idx_split1:idx_split2]
    test = dataset[idx_split2:] 

    return [ training, validation, test ]


# Extract training and test set
print('Setup Dataset')
rerun = False

saveImgEdgePath = folderpath+'pickleImgEdge'
saveImgLinePath = folderpath+'pickleImgLine'
saveImgCornerPath = folderpath+'pickleImgCorner'
saveImgStatsPath = folderpath+'pickleImgStats'
saveImgRawPath = folderpath+'pickleImgRaw'

saveImgEdgePathSubmission = folderpath+'pickleImgEdgeSubmission'
saveImgLinePathSubmission = folderpath+'pickleImgLineSubmission'
saveImgCornerPathSubmission = folderpath+'pickleImgCornerSubmission'
saveImgStatsPathSubmission = folderpath+'pickleImgStatsSubmission'
saveSubmissionImgRawPath = folderpath+'pickleImgRawSubmission'

if rerun == True or not os.path.isfile(saveImgRawPath):
    train_ImgEdge = getEdges(train, train_path)
    train_ImgLine = getLines(train, train_path)
    train_ImgCorner = getCorners(train, train_path)
    train_ImgStats = extract_features(train, train_path)
    # train_ImgRaw = get_raw(train, train_path)

    pickle.dump(train_ImgEdge, open( saveImgEdgePath , 'wb'))
    pickle.dump(train_ImgLine, open( saveImgLinePath , 'wb'))
    pickle.dump(train_ImgCorner, open( saveImgCornerPath , 'wb'))
    pickle.dump(train_ImgStats, open( saveImgStatsPath , 'wb'))
    # pickle.dump(train_ImgRaw, open( saveImgRawPath , 'wb'))

    submission_ImgEdge = getEdges(test, test_path)
    submission_ImgLine = getLines(test, test_path)
    submission_ImgCorner = getCorners(test, test_path)
    submission_ImgStats = extract_features(test, test_path)
    # submission_ImgRaw = get_raw(test, test_path)

    pickle.dump(submission_ImgEdge, open( saveImgEdgePathSubmission , 'wb'))
    pickle.dump(submission_ImgLine, open( saveImgLinePathSubmission , 'wb'))
    pickle.dump(submission_ImgCorner, open( saveImgCornerPathSubmission , 'wb'))
    pickle.dump(submission_ImgStats, open( saveImgStatsPathSubmission , 'wb'))
    # pickle.dump(submission_ImgRaw, open( saveSubmissionImgRawPath , 'wb'))

else:
    train_ImgEdge = pickle.load(open(saveImgEdgePath, 'rb'))
    train_ImgLine = pickle.load(open(saveImgLinePath, 'rb'))
    train_ImgCorner = pickle.load(open(saveImgCornerPath, 'rb'))
    train_ImgStats = pickle.load(open(saveImgStatsPath, 'rb'))
    # train_ImgRaw = pickle.load(open(saveImgRawPath, 'rb'))

    submission_ImgEdge = pickle.load(open(saveImgEdgePathSubmission, 'rb'))
    submission_ImgLine = pickle.load(open(saveImgLinePathSubmission, 'rb'))
    submission_ImgCorner = pickle.load(open(saveImgCornerPathSubmission, 'rb'))
    submission_ImgStats = pickle.load(open(saveImgStatsPathSubmission, 'rb'))
    # submission_ImgRaw = pickle.load(open(saveSubmissionImgRawPath, 'rb'))

X = np.hstack((train_ImgEdge, train_ImgLine, train_ImgCorner, train_ImgStats))
# X = np.hstack((train_ImgEdge, train_ImgLine, train_ImgCorner))

print('Setup Dataset Labels')
y_train = []

# flatten = lambda l: [item for sublist in l for item in sublist]
# labels = np.array(list(set(flatten([l.split(' ') for l in train['tags'].values]))))
labels = np.array(['clear', 'partly_cloudy', 'cloudy', 'haze', 'primary', 'water', 'bare_ground', 
    'agriculture', 'cultivation', 'habitation', 'road', 'conventional_mine', 'artisinal_mine', 
    'selective_logging', 'slash_burn', 'blooming', 'blow_down'])

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for tags in tqdm(train.tags.values, mininterval=10):
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    y_train.append(targets)
    
y = np.array(y_train, np.float32)

# A training array of each statistic for each image as shown in extract_features()
print('X.shape = ' + str(X.shape))
# A training array of 0 or 1s to represent label
print('y.shape = ' + str(y.shape))

# Splitting training set into training and validation set
print('Training for validation and test set predictions')
train_dataset, valid_dataset, test_dataset = splitSet(X, 0.6, 0.8)
train_labels, valid_labels, test_labels = splitSet(y, 0.6, 0.8)

clf = RandomForestClassifier(n_estimators=100)

clf = clf.fit(train_dataset, train_labels)


print('Validation Set Weighted F1 Scores')
validation_prediction = np.array( [ clf.predict(test_chip.reshape(1,-1))[0] for test_chip in tqdm(valid_dataset, mininterval=10) ] ).astype(int)
f1_validation = f1_score(valid_labels, validation_prediction, average='weighted' )
print(f1_validation)


print('Test Set Weighted F1 Scores')
test_prediction = np.array( [ clf.predict(test_chip.reshape(1,-1))[0] for test_chip in tqdm(test_dataset, mininterval=10) ] ).astype(int)
f1_test = f1_score(test_labels, test_prediction, average='weighted' )
print(f1_test)


# # Making Final Predictions using all training data
# print('Fitting model')
# clf = clf.fit(X, y)

# print('Making submission predictions')
# submissionSet = np.hstack((submission_ImgEdge, submission_ImgLine, submission_ImgCorner, submission_ImgStats))
# submissionPredictions = [ clf.predict(test_chip.reshape(1,-1)) for test_chip in tqdm(submissionSet, mininterval=10) ]
# predictionLabels = [' '.join(labels[row[0] > 0.2]) for row in submissionPredictions]

# #Outputting predictions to csv
# subm = pd.DataFrame()
# subm['image_name'] = test.image_name.values
# subm['tags'] = predictionLabels
# subm.to_csv(folderpath+'submission.csv', index=False)


# Analytics
def getLabelDistribution(labels, labelNameArray):
    labelCount = [ np.sum(labels[:,i]) for i in range(0, len(labels[0])) ]
    labelNameCount = {key: val for key, val in zip(labelNameArray, labelCount)}

    return labelNameCount, labelCount

def getPrecision(labels, predictions):
    Tp = 0
    Fp = 0
    for label, prediction in zip(labels, predictions):
        if label==1 and prediction==1:
            Tp += 1
        if label==0 and prediction==1:
            Fp += 1

    if Tp+Fp == 0:
        return 1

    return (Tp / ( Tp + Fp ))

def getRecall(labels, predictions):
    Tp = 0
    Fn = 0
    for label, prediction in zip(labels, predictions):
        if label==1 and prediction==1:
            Tp += 1
        if label==1 and prediction==0:
            Fn += 1

    if Tp+Fn == 0:
        return 1

    return (Tp / ( Tp + Fn ))

def getPrecisionRecall(labels, predictions, labelNames):
    precision = [ getPrecision(labels[:, col], predictions[:, col]) for col in range(0, len(labels[0])) ]
    recall = [ getRecall(labels[:, col], predictions[:, col]) for col in range(0, len(labels[0])) ]
    precision = np.array(precision)
    recall = np.array(precision)
    # npPR = np.vstack((precision, recall))
    # npPR = npPR.transpose()

    labelPR = {labelName: (precision[idx], recall[idx]) for idx, labelName in enumerate(labelNames)}

    return labelPR, precision, recall

_, labelCounts = getLabelDistribution(test_labels, labels)
labelPercentage = np.array( [ np.array([ count / np.sum(labelCounts) ]) for count in labelCounts ] )
_, precision, recall = getPrecisionRecall(test_labels, test_prediction, labels)

fig, ax = plt.subplots()
index = np.arange(len(labels))
bar_width = 0.25
opacity = 0.8
 
rects1 = plt.bar(index, precision, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Precision')
 
rects2 = plt.bar(index + bar_width, recall, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Recall')

rects2 = plt.bar(index + bar_width + bar_width, labelPercentage, bar_width,
                 alpha=opacity,
                 color='y',
                 label='percentage')
 
plt.xlabel('Label')
plt.ylabel('Scores')
plt.title('Scores by Label')
plt.xticks(rotation=70)
plt.xticks(index + bar_width, (label for label in labels))
plt.legend()
 
plt.tight_layout()
plt.show()


# Labels 
# ['selective_logging', 'conventional_mine', 'partly_cloudy',
#        'artisinal_mine', 'haze', 'slash_burn', 'primary', 'clear',
#        'bare_ground', 'blooming', 'water', 'road', 'cloudy', 'habitation',
#        'agriculture', 'blow_down', 'cultivation']

# Training set label distribution
# {'slash_burn': 209.0, 'blooming': 332.0, 'water': 7262.0, 'cloudy': 2330.0, 'selective_logging': 340.0,
#  'road': 8076.0, 'primary': 37840.0, 'clear': 28203.0, 'haze': 2695.0, 'agriculture': 12338.0, 'cultivation': 4477.0, 
#  'partly_cloudy': 7251.0, 'bare_ground': 859.0, 'conventional_mine': 100.0, 'artisinal_mine': 339.0, 
#  'habitation': 3662.0, 'blow_down': 98.0}

