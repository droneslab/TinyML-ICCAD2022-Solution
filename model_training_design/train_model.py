

import numpy as np
import pandas as pd

from help_code_demo import ToTensor, IEGM_DataSET
import time

import argparse

import torchvision.transforms as transforms


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 
from sklearn.metrics import fbeta_score 
from sklearn.metrics import confusion_matrix


def extract_features_for_single_peak_optimized_v1(x_peaks):
    peak_features_count = len(x_peaks)
    diff_intervals = [x - x_peaks[i - 1] for i, x in enumerate(x_peaks)][1:]

    if len(diff_intervals) >1:
        peak_features_min_int = np.min(diff_intervals)
        peak_features_max_int = np.max(diff_intervals)
        peak_features_avg_int = np.ceil(np.average(diff_intervals))
        
    else:
        peak_features_min_int = 1
        peak_features_max_int = 1
        peak_features_avg_int = 1


    
    feature_list=[peak_features_count, peak_features_min_int, peak_features_max_int,peak_features_avg_int]
   
    return feature_list

def supress_non_maximum(peak_indices, X_data, window = 30):

    if len(peak_indices)<1:
        return []

    new_peak_indices=[]
    last_peak=peak_indices[0]
    for i in range(1, len(peak_indices)):
        curr_diff = peak_indices[i] - last_peak
        if curr_diff > window:
            new_peak_indices.append(last_peak)
            last_peak = peak_indices[i]
        else:
            if X_data[peak_indices[i]] > X_data[last_peak]:
                last_peak = peak_indices[i]
    if len(new_peak_indices)==0:
        return new_peak_indices
    if new_peak_indices[-1] != last_peak :
        new_peak_indices.append(last_peak)
        
    return new_peak_indices




def extract_peaks_features_optimized_v1(X_data, std_val=1.8, window=38):
    X_data_new = np.array(X_data)
    std_arr = np.abs(np.std(X_data_new)*std_val)
    peak_indices =np.where(np.abs(X_data_new) > std_arr)[0]


    peak_indices = supress_non_maximum(peak_indices, X_data, window)

    peaks_features = extract_features_for_single_peak_optimized_v1(peak_indices)

    return peaks_features

def extract_features_extended(X_data, sigma=1.8, window=38):
    X_data_peak_features = []
    for i in range(len(X_data)):
        X_data_features = extract_peaks_features_optimized_v1(X_data[i], sigma, window)
        X_data_peak_features.append(X_data_features)
  
    X_data_peaks_feat_df = pd.DataFrame.from_dict(X_data_peak_features)

    return X_data_peaks_feat_df


def extract_labels(dataset):
    dataset_labels = []
    for itm in dataset.names_list:
        dataset_labels.append(itm.split(' ')[0].split('-')[1])
    return dataset_labels

def load_dataset(dataset):
    X_data = []
    y_data = []
    for i in dataset:
        iegm_seg = i['IEGM_seg'].flatten()
        label = i['label']
        X_data.append(iegm_seg)
        y_data.append(label)
    return np.array(X_data), np.array(y_data).reshape(-1,1), np.array(extract_labels(dataset))

def show_validation_results(C, total_time):
        #C = C_board
        print(C)

        #total_time = 0#sum(timeList)
        avg_time = total_time#np.mean(timeList)
        acc = (C[0][0] + C[1][1]) / (C[0][0] + C[0][1] + C[1][0] + C[1][1])
        precision = C[1][1] / (C[1][1] + C[0][1])
        sensitivity = C[1][1] / (C[1][1] + C[1][0])
        FP_rate = C[0][1] / (C[0][1] + C[0][0])
        PPV = C[1][1] / (C[1][1] + C[1][0])
        NPV = C[0][0] / (C[0][0] + C[0][1])
        F1_score = (2 * precision * sensitivity) / (precision + sensitivity)
        F_beta_score = (1+2**2) * (precision * sensitivity) / ((2**2)*precision + sensitivity)

        print("\nacc: {},\nprecision: {},\nsensitivity: {},\nFP_rate: {},\nPPV: {},\nNPV: {},\nF1_score: {}, "
                "\ntotal_time: {},\n average_time: {}".format(acc, precision, sensitivity, FP_rate, PPV, NPV, F1_score,
                                                        total_time, avg_time))

        print("F_beta_score : ", F_beta_score)

def main(args):
    # Hyperparameters
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices

    hyper_param = {}
    hyper_param["sigma"] = args.sigma
    hyper_param["window"] = args.window
    hyper_param["depth"] = args.depth
    #sigma = args.sigma
    #window = args.window


    print("Hyperparameters: ")
    print(hyper_param)

    
    # Start dataset loading
    trainset = IEGM_DataSET(root_dir=path_data,
                            indice_dir=path_indices,
                            mode='train',
                            size=SIZE,
                            transform=transforms.Compose([ToTensor()]))

    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='test',
                           size=SIZE,
                           transform=transforms.Compose([ToTensor()]))


    X_test, y_test, _ = load_dataset(testset)
    X_train, y_train, _ = load_dataset(trainset)

    start_time = time.time()

    X_train_peaks_feat_extend_df = extract_features_extended(X_train, hyper_param["sigma"], hyper_param["window"])
    X_test_peaks_feat_extend_df = extract_features_extended(X_test, hyper_param["sigma"], hyper_param["window"])    


    DTC_1 = DecisionTreeClassifier(criterion='entropy', max_depth=hyper_param["depth"], random_state=0)

    DTC_1.fit(X_train_peaks_feat_extend_df, y_train)

    #LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X_train_peaks_feat_extend_df, y_train)

    print("========================================")

    print("Train:")
    train_pred = DTC_1.predict(X_train_peaks_feat_extend_df)
    C_DT_train = confusion_matrix(y_train, train_pred)
    print(C_DT_train)
    print("False Positives % : " ,100.0*(C_DT_train[0][1]/(C_DT_train[1][1]+C_DT_train[0][1])))
    print("False Negative % : " ,100.0*(C_DT_train[1][0]/(C_DT_train[1][0]+C_DT_train[0][0])))    
    cur_f_beta = fbeta_score(y_train, train_pred, beta=2) 
    print(f'F_beta score: {cur_f_beta}') 

    print("test:")
    test_pred = DTC_1.predict(X_test_peaks_feat_extend_df)
    C_DT_test = confusion_matrix(y_test, test_pred)
    print(C_DT_test)
    print("False Positives % : " ,100.0*(C_DT_test[0][1]/(C_DT_test[1][1]+C_DT_test[0][1])))
    print("False Negative % : " ,100.0*(C_DT_test[1][0]/(C_DT_test[1][0]+C_DT_test[0][0])))     
    cur_f_beta = fbeta_score(y_test, DTC_1.predict(X_test_peaks_feat_extend_df), beta=2) 
    print(f'F_beta score: {cur_f_beta}') 

    print("Decision Tree")
    #print("Best Intercept" , LR.intercept_, "Best coeff", LR.coef_)
    # https://mljar.com/blog/extract-rules-decision-tree/ 
    # get the text representation
    text_representation = tree.export_text(DTC_1)
    print(text_representation)
    print("========================================")
    
    total_time = (time.time() - start_time)
    print("Train score", round(DTC_1.score(X_train_peaks_feat_extend_df, y_train), 4))
    print("Test score", round(DTC_1.score(X_test_peaks_feat_extend_df, y_test), 4))

    y_test_pred = DTC_1.predict(X_test_peaks_feat_extend_df)
    print()
    C_DT = confusion_matrix(y_test, y_test_pred)
    show_validation_results(C_DT, total_time)    


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='../../../tinyml_contest_data_training/')
    argparser.add_argument('--path_indices', type=str, default='../../../data_indices')
    argparser.add_argument('--sigma', type=float, default=1.8)
    argparser.add_argument('--window', type=int, default=40)
    argparser.add_argument('--depth', type=int, default=1)

    args = argparser.parse_args()

    start_time = time.time()
    
    main(args)

    total_time = (time.time() - start_time)
    
    print("--- %s seconds ---" % total_time)
