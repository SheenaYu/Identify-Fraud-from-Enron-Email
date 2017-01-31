"""
Author: Sheena Yu

This file contains all functions used to explore/clean the Enron dataset and 
tune supervised classification algorithms.
"""






def distribution_poi_nonpoi(data_dict):
    """
    Function to get the number of pois and non-pois in the data_dict.
    """
    poi_counter = 0
    nonpoi_counter = 0
    for individual in data_dict:
        if data_dict[individual]["poi"]:
            poi_counter += 1
        else:
            nonpoi_counter += 1
    return poi_counter, nonpoi_counter



def count_nans_by_person(data_dict, num_person):
    """
    Function to count NaN values in each data point and sort them in descending order;
    Args:
        data_dict: Insiders and their financial and email information
        num_person: Top number of data points with highest occurrences of NaN values
    Returns:
        A list of tuples with person and their frequency of NaN values
    """
    person_nan = []
    for individual in data_dict:
        nan_counter = 0
        for v in data_dict[individual].itervalues():
            if v == "NaN":
                nan_counter += 1
        person_nan.append((individual, nan_counter))
    person_nan = sorted(person_nan, key=lambda x:x[1], reverse=True)[:num_person]
    return person_nan



def count_non_nans_by_feature(data_dict, features_list):
    """
    Function to count non-nan values for each feature.
    """
    feature_nan = {}
    for feature in features_list:
        for k, v in data_dict.iteritems():
            if v[feature] != 'NaN':
                feature_nan[feature] = feature_nan.get(feature, 0) + 1
    return feature_nan



def get_outliers_via_boxplot(data_pd, feature, threshold):
    """
    Function to extract outliers based on different features
    Args:
        data_pd: Reformattd data_dict, pandas dataframe
        feature: Interesting feature that can be used to identify outliers
        threshold: Numerical value of that feature that is used to seperate non-outliers and outliers
    Returns:
        List of outliers
    """
    outliers = data_pd[["poi", feature]][data_pd[feature] > threshold]
    return outliers




# Develop get_features function
def get_features(data_dict, remove_poi=True, remove_email_address=True):
    """
    Function to extract all the keys (mostly features) from each person's dictionary value
    Args:
        data_dict: Insiders and their financial and email information
        remove_poi: Whether to keep "poi" in the result list
        remove_email_address: Whether to keep "email_address" in the result list
    Returns:
        A list containing needed keys/features
    """
    features_list = []
    # Focus on one data point is enough to get all needed features
    person = data_dict["SKILLING JEFFREY K"]
    for key in person.keys():
        if remove_poi and not remove_email_address:
            if key != "poi":
                features_list.append(key)
        elif remove_email_address and not remove_poi:
            if key != "email_address":
                features_list.append(key)
        elif remove_poi and remove_email_address:
            if key != "poi" and key != "email_address":
                features_list.append(key)
        else:
            features_list.append(key)
    return features_list



def detect_inconsistent_data(data_dict):
    """
    Function to detect inconsistent data points;
    Args:
        data_dict: Insiders and their financial and email information
    Returns:
        Persons with inconsistent information
    """
    payments_features = ['salary', 'deferral_payments', 'loan_advances', 'bonus', \
                         'deferred_income', 'expenses', 'other', \
                         'long_term_incentive', 'director_fees', 'total_payments']

    stock_features = ['exercised_stock_options', 'restricted_stock', \
                      'restricted_stock_deferred', 'total_stock_value']

    features_combo = [("payments", payments_features), ("stocks", stock_features)]
    
    results = []
    for p in data_dict:
        for feat_set in features_combo:
            combination = 0
            for feat in feat_set[1][:-1]:
                if data_dict[p][feat] != "NaN":
                    combination += data_dict[p][feat]

            original = data_dict[p][feat_set[1][-1]]
            if original=="NaN":
                original = 0
            if combination != original:
                results.append({"features_type": feat_set[0], "person":p, "combination_value":combination, "original_value": original})
    return results



def fix_inconsistent_data(data_dict):
    """
    Function to fix inconsistent data points;
    Args:
        data_dict: Insiders and their financial and email information
    Returns:
        corrected data_dict
    """
    # Replace the wrong values of "BELFER ROBERT"--shift to the right
    shifted_features = ['salary', 'bonus', 'long_term_incentive', 'deferred_income', 'deferral_payments', 'loan_advances', 'other', 'expenses', 'director_fees', 'total_payments', 'exercised_stock_options', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value']

    i = 0
    while i < len(shifted_features)-1:
        BR = data_dict["BELFER ROBERT"]
        BR[shifted_features[i]] = BR[shifted_features[i+1]]
        i += 1

    total_stock_value = 0
    for feat in shifted_features[-4:-1]:
        BR = data_dict["BELFER ROBERT"]
        if BR[feat]=="NaN":
            continue
        total_stock_value += BR[feat]
        if total_stock_value == 0:
            total_stock_value = "NaN"

    data_dict["BELFER ROBERT"][shifted_features[-1]] = total_stock_value

    # Replace the wrong values of "BHATNAGAR SANJAY"--shift to the left
    j = len(shifted_features)-1
    while j > 0:
        BS = data_dict["BHATNAGAR SANJAY"]
        BS[shifted_features[j]] = BS[shifted_features[j-1]]
        j -= 1
        
    data_dict["BHATNAGAR SANJAY"]["salary"] = "NaN"



## Create poi_pctg feature
def add_poi_pctg_feature(data_dict, features_list):
    """
    Function to add new poi_pctg feature to both features_list and data_dict
    Args:
       data_dict: Insiders and their financial and email information
       features_list: features of each person excluding email_address and poi
    """
    email_features = ['from_messages', 'to_messages', 'from_poi_to_this_person', 'from_this_person_to_poi']
    for p in data_dict:
        person = data_dict[p]
        
        email_list = []
        for feat in email_features:
            email_list.append(person[feat])
        
        if "NaN" in email_list:
            poi_pctg = "NaN"
        else:
            poi_pctg = float(sum(email_list[2:])) / float(sum(email_list[:2]))
            
        person["poi_pctg"] = poi_pctg
    features_list += ["poi_pctg"]


    
## Create controllable_gain feature, which combine features prone to manipulation together
def add_controllable_gain_feature(data_dict, features_list):
    """
    Function to add new controllable_gain feature to both features_list and data_dict
    Args:
       data_dict: Insiders and their financial and email information
       features_list: features of each person excluding email_address and poi
    """
    controllable_gain_features = ['salary', 'bonus', 'loan_advances', 'other',\
                                 'expenses', 'director_fees', 'exercised_stock_options']
    for p in data_dict:
        person = data_dict[p]
        controllable_gain = 0
        for feat in controllable_gain_features:
            if person[feat] == "NaN":
                continue
            controllable_gain += person[feat]
        person["controllable_gain"] = controllable_gain
    features_list += ["controllable_gain"]

###########################################################################################################################################

# tune classifier and obtain scores based on parameters above
def evaluate_algorithms(my_dataset, features_list, classifier, parameters):
    """
    Function to tune classifier and get evaluation metric scores based on the best combination of parameters
    Args:
        scaler: feature scaler
        sbk: feature selection object
        classifier: The classifier/algorithm to test
        parameters: Dictionary of different parameters and their list of values
    Returns:
        Classifier's results
    """

    from feature_format import featureFormat, targetFeatureSplit
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import GridSearchCV
    from tester import dump_classifier_and_data   
    from tester import test_classifier

    ## Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    ## Create scaler and selection object
    scaler = MinMaxScaler()
    skb = SelectKBest()
    steps = [('scaling',scaler), ('SKB', skb), ('algo', classifier)]

    pipeline = Pipeline(steps)
    sss = StratifiedShuffleSplit(n_splits=100, test_size=0.2, train_size=None, random_state=42)
    gs = GridSearchCV(pipeline, parameters, n_jobs = 1, cv=sss, scoring="f1")
    gs.fit(features, labels)
    clf = gs.best_estimator_
    features_selected = [features_list[i+1] for i in clf.named_steps['SKB'].get_support(indices=True)]
    print '\nClassifier:', classifier 
    print '\nThe features selected by SelectKBest:\n', features_selected
    print "\nThe best parameters:\n", gs.best_params_, '\n' 
    # Dump your classifier, dataset, and features_list so anyone can
    # check your results. You do not need to change anything below, but make sure
    # that the version of poi_id.py that you submit can be run on its own and
    # generates the necessary .pkl files for validating your results.
    from tester import dump_classifier_and_data
    dump_classifier_and_data(clf, my_dataset, features_list)
    print "Tester Classification report\n" 
    test_classifier(clf, my_dataset, features_list)
    return clf

