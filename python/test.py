# ctypes_test.py
import random
import time

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from cgsvm import CgSvm

# FULL_PROBLEM = Fullproblem(
#     n=20,
#     p=2,
#     C=0.5,
#     alpha=,
#     beta=,
#     active=,
#     inactive=,
#     partialH=
# )

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_cgsvm_cancer_data():
    breast_cance_bunch = load_breast_cancer(as_frame=True)
    breast_cance_df = breast_cance_bunch['data']
    breast_cance_df['target'] = breast_cance_bunch['target']
    breast_cance_df.sort_values(by=['target'], inplace=True, ascending=False)
    n_positive = breast_cance_df['target'].sum()
    breast_cance_df.drop(labels= 'target', axis=1, inplace=True)
    return breast_cance_df, n_positive

def get_sklearn_cancer_data():
    breast_cance_bunch = load_breast_cancer(as_frame=True)
    return breast_cance_bunch['data'].fillna(0), breast_cance_bunch['target']

def print_sklearn_results():
    for i in range(len(skl_preds)):
        if ( skl_preds[i] > 0.5 and bc_df_target[i] > 0.5) or ( skl_preds[i] < 0.5 and bc_df_target[i] < 0.5):
            print(f'{bcolors.OKGREEN}{bc_np[i][0]} {predictions[i]}{bcolors.ENDC}')
            correct_sklearn_count += 1

def print_cg_results(predictions, n_positive):
    for i in range(len(predictions)):
        if (i<n_positive and predictions[i]>0) or (i>=n_positive and predictions[i]<0):
            print(f'{bcolors.OKGREEN}{bc_np[i][0]} {predictions[i]}{bcolors.ENDC}')
            correct_cg_count += 1
        else:
            print(f'{bcolors.FAIL}{bc_np[i][0]} {predictions[i]}{bcolors.ENDC}')

def main():
    # Load the shared library into ctypes
    cg_svm = CgSvm()
    svc = SVC()

    bc_df, n_positive = get_cgsvm_cancer_data()
    bc_df_sk, bc_df_target = get_sklearn_cancer_data()

    skl_start_time = time.time()
    svc_model = svc.fit(bc_df_sk, bc_df_target)
    skl_end_time = time.time()
    skl_preds = svc_model.predict(bc_df_sk)

    cg_svm_model = cg_svm.fit(bc_df, n_positive, save_to_file=True, file_name='bc_pythontest.txt')
    predictions = cg_svm_model.transform(bc_df)

    print('cgsvm time (microseconds):', cg_svm_model.trained_model.trainElapsedTime)
    print('sklearn time (microseconds):', round(1000000*(skl_end_time-skl_start_time)))

    correct_cg_count = 0
    for i in range(len(predictions)):
        if (i<n_positive and predictions[i]>0) or (i>=n_positive and predictions[i]<0):
            correct_cg_count += 1

    print('cgsvm accuracy: ', correct_cg_count/bc_df.shape[0])
    print('sklearn accuracy: ', accuracy_score(bc_df_target, skl_preds)  )

    scatter_x = np.array([0.0, 0.0, 3.0, 4.0])
    scatter_y = np.array([0.0, 1.0, 3.0, 5.0])
    df = pd.DataFrame({'x':scatter_x, 'y':scatter_y})
    cg_svm_model = cg_svm.fit(df, 5, save_to_file=True, file_name='imagegen_pythontest.txt')



if __name__ == "__main__":
    main()
