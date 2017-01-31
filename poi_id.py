#!/usr/bin/python
# Created by Sheena Yu

import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import pprint
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import enron_func

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

############################################################################################################

# Investigate the dataset
print "Total number of data points: ", len(data_dict)
total_features = len(data_dict["SKILLING JEFFREY K"])
print "\nTotal number of features: ", total_features
print "\nExample of one data point in the dictionary: \n", data_dict["SKILLING JEFFREY K"]

# Allocation across classes (POI / non-POI)
poi_counter, nonpoi_counter = enron_func.distribution_poi_nonpoi(data_dict)	
string = "\nThere are {0:d} persons of interest in the dataset, and {1:d} innocent persons.\n"
print string.format(poi_counter, nonpoi_counter)

# ############################################################################################################
# Missing values, outliers and odd data points


# Missing values by persons
## Find top 5 data points with the most "NaN" values in the value dictionary
top_10_nan_persons = enron_func.count_nans_by_person(data_dict, 5)
print top_10_nan_persons

## Remove the data point with 20 "NaN" features
data_dict.pop("LOCKHART EUGENE E", 0)
## Remove data point "THE TRAVEL AGENCY IN THE PARK"
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

print "Length of data now is: ", len(data_dict)

# Missing values by features
## Count all non-nan values of each feature
full_features_list = enron_func.get_features(data_dict, remove_poi=False, remove_email_address=False)
feature_non_nan_dict = enron_func.count_non_nans_by_feature(data_dict, full_features_list)
pprint.pprint(feature_non_nan_dict)


# Outliers

## After inspect the salary boxplot, we identified a outlier "total", remove total
data_dict.pop('TOTAL', 0)

## Then we regenerate the salary by poi boxplot 
data_dict_pd = [{k: data_dict[p][k] if data_dict[p][k] != 'NaN' else None for k in data_dict[p].keys()} for p in data_dict.keys()]
data_pd = pd.DataFrame(data_dict_pd)

data_pd.boxplot("salary", by="poi")
data_pd.boxplot("bonus", by="poi")
data_pd.boxplot("total_payments", by="poi")
data_pd.boxplot("total_stock_value", by="poi")
# plt.show()

salary_outliers = enron_func.get_outliers_via_boxplot(data_pd, "salary", 1000000)
bonus_outlier2 = enron_func.get_outliers_via_boxplot(data_pd, "bonus", 5000000)
tp_outlier = enron_func.get_outliers_via_boxplot(data_pd, "total_payments", 100000000)
ts_outliers = enron_func.get_outliers_via_boxplot(data_pd, "total_stock_value", 25000000)

"""
There are one data point with salary greater than 1m in the non-poi group and two in the poi group
Two potential poi outliers are Lay Kenneth and Skilling Jeffrey, the other non-poi outlier is Frevert Mark. 
bonus_outliers: non-poi: LAVORATO, JOHN J; poi: LAY, KENNETH L, BELDEN, TIMOTHY N, SKILLING, JEFFREY K
tp_outlier: LAY, KENNETH L
ts_outlier: LAY, KENNETH L, SKILLING, JEFFREY K, HIRKO, JOSEPH
"""

# Inconsistent Data Points
odd_data = enron_func.detect_inconsistent_data(data_dict)
print odd_data

enron_func.fix_inconsistent_data(data_dict)

# ############################################################################################################

# Feature Engineering and Feature Selection

features_list = enron_func.get_features(data_dict)
print len(features_list)

enron_func.add_poi_pctg_feature(data_dict, features_list)
enron_func.add_controllable_gain_feature(data_dict, features_list)

# Verify that the two new features are successfully added to data_dict
pprint.pprint(data_dict["SKILLING JEFFREY K"])

# Add target label to features_list
features_list = ["poi"] + features_list

# Target and features (including two new engineered features)
print features_list

# ############################################################################################################

# Try a varity of classifiers, tune classifiers to achieve better than .3 precision and recall 

## Copy data_dict to my_dataset for easy export
my_dataset = data_dict


## create the classifiers
lg = LogisticRegression()
gnb = GaussianNB()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
knn = KNeighborsClassifier()
svc = SVC()

## create parameters for classifiers to be used in pipeline
k_range = range(3,13)

params_lg = {
    'SKB__k' : k_range,
    'algo__tol' : [10**-5, 10**-4, 10**-3, 10**-2, 1, 10, 100, 1000],
    'algo__C' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

params_gnb = {
        'SKB__k' : k_range
        }

params_dt = {
        'SKB__k' : k_range,
        'algo__min_samples_split' : [2, 4, 6, 8, 10, 15, 20, 25, 30],
        'algo__criterion' : ['gini', 'entropy']
        }

params_rf = {
    'SKB__k' : k_range,
    'algo__criterion' : ['gini', 'entropy'],
    'algo__min_samples_split' : [2, 4, 6, 8, 10]
}

params_knn = {
        'SKB__k' : k_range,
        'algo__n_neighbors' : range(2, 10),
        'algo__algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto']
        }
    
params_svm = {
        'SKB__k' : k_range,
        'algo__kernel' : ['rbf'],
        'algo__C': [0.1, 1, 2, 4, 6, 8, 10], 
        'algo__gamma' : [0.01, 0.1, 1, 10.0, 50.0, 100.0],
        "algo__tol":[10**-1, 10**-10]
        }


# # run the dataset through each of the six classifiers 

# # logisticRegression Accuracy: 0.86207	Precision: 0.44812	Recall: 0.14900	F1: 0.22364	F2: 0.17196
# enron_func.evaluate_algorithms(my_dataset, features_list, lg, params_lg)

# # Accuracy: 0.86460	Precision: 0.48707	Recall: 0.29200	F1: 0.36511	F2: 0.31743
# enron_func.evaluate_algorithms(my_dataset, features_list, rf, params_rf)

# # DecisionTree Accuracy: 0.85027	Precision: 0.43856	Recall: 0.43900	F1: 0.43878	F2: 0.43891
enron_func.evaluate_algorithms(my_dataset, features_list, dt, params_dt)

# # knn Accuracy: 0.86200	Precision: 0.46377	Recall: 0.22400	F1: 0.30209	F2: 0.24983
# enron_func.evaluate_algorithms(my_dataset, features_list, knn, params_knn)

# # svm Accuracy: 0.87607	Precision: 0.60383	Recall: 0.20500	F1: 0.30608	F2: 0.23620
# enron_func.evaluate_algorithms(my_dataset, features_list, svc, params_svm)

# Accuracy: 0.85013	Precision: 0.41218	Recall: 0.29100	F1: 0.34115	F2: 0.30918
# enron_func.evaluate_algorithms(my_dataset, features_list, gnb, params_gnb)