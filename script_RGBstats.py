import numpy as np
import os
import pandas as pd
import random
from tqdm import tqdm

import scipy
from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

from PIL import Image

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)

# Load data
folderpath = '/Users/cloudlife/GitHub/kaggleplanet/'
train_path = folderpath+'train-jpg/'
test_path = folderpath+'test-jpg/'
train = pd.read_csv(folderpath+'train.csv')
test = pd.read_csv(folderpath+'sample_submission.csv')

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

    for image_name in tqdm(im_features.image_name.values, miniters=100): 
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
    
    return im_features

# Extract features
print('Extracting train features')
train_features = extract_features(train, train_path)
print('Extracting test features')
test_features = extract_features(test, test_path)

# Prepare data
X = np.array(train_features.drop(['image_name', 'tags'], axis=1))
y_train = []

flatten = lambda l: [item for sublist in l for item in sublist]
labels = np.array(list(set(flatten([l.split(' ') for l in train_features['tags'].values]))))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for tags in tqdm(train.tags.values, miniters=1000):
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    y_train.append(targets)
    
y = np.array(y_train, np.uint8)

# A training array of each statistic for each image as shown in extract_features()
print('X.shape = ' + str(X.shape))
# A training array of 0 or 1s to represent label
print('y.shape = ' + str(y.shape))

# Splitting training set into training and validation set
idx_split1 = int( len(X) * 0.60) # splits training into 60% training, 20% validation, and 20% test set
idx_split2 = int( len(X) * 0.80) # splits training into 60% training, 20% validation, and 20% test set

X_training = X[: idx_split1]
y_training = y[: idx_split1]

X_validation = X[idx_split1: idx_split2]
y_validation = y[idx_split1: idx_split2]

X_test = X[idx_split2:]
y_test = y[idx_split2:]

clf = RandomForestClassifier(n_estimators=100)

clf = clf.fit(X_training, y_training)

print('Validation Set Weighted F1 Scores')
y_validation_prediction = [ clf.predict(test_chip.reshape(1,-1))[0] for test_chip in tqdm(X_validation) ]
f1_validation = f1_score(y_validation, np.array(y_validation_prediction).astype(int), average='weighted' )
print(f1_validation)


print('Test Set Weighted F1 Scores')
y_test_prediction = [ clf.predict(test_chip.reshape(1,-1))[0] for test_chip in tqdm(X_test) ]
f1_test = f1_score(y_test, np.array(y_test_prediction).astype(int), average='weighted' )
print(f1_test)



# # Making Final Predictions using all training data
# clf = clf.fit(X, y)

# X_predictions = np.array(test_features.drop(['image_name', 'tags'], axis=1))
# y_predictions = [ clf.predict(test_chip.reshape(1,-1)) for test_chip in tqdm(X_predictions) ]

# preds = [' '.join(labels[y_pred_row[0] > 0.2]) for y_pred_row in y_predictions]

# #Outputting predictions to csv
# subm = pd.DataFrame()
# subm['image_name'] = test_features.image_name.values
# subm['tags'] = preds
# subm.to_csv(folderpath+'submission.csv', index=False)
