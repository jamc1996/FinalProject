# ctypes_test.py
import random

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

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

def get_break_cancer_data():
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

def main():
    # Load the shared library into ctypes
    cg_svm = CgSvm()
    bc_df, n_positive = get_break_cancer_data()
    bc_df_sk, bc_df_target = get_sklearn_cancer_data()

    svc = SVC()
    scvfit = svc.fit(bc_df_sk, bc_df_target)
    skl_preds = scvfit.predict(bc_df_sk)
    print(skl_preds)
    print(n_positive)

    bc_np = bc_df.to_numpy(dtype=np.float64, na_value=0.0)
    print(bc_np.shape)
    svm_model = cg_svm.fit(bc_np, n_positive, save_to_file=True, file_name='pythontest_bc.txt')

    predictions = svm_model.transform(bc_np)
    print(svm_model.trained_model.trainElapsedTime)
    print(len(predictions))
    correct_cg_count = 0
    for i in range(len(predictions)):
        if (i<n_positive and predictions[i]>0) or (i>=n_positive and predictions[i]<0):
            print(f'{bcolors.OKGREEN}{bc_np[i][0]} {predictions[i]}{bcolors.ENDC}')
            correct_cg_count += 1
        else:
            print(f'{bcolors.FAIL}{bc_np[i][0]} {predictions[i]}{bcolors.ENDC}')
    print(100*correct_cg_count/bc_np.shape[0])

    correct_sklearn_count = 0
    for i in range(len(skl_preds)):
        if ( skl_preds[i] > 0.5 and bc_df_target[i] > 0.5) or ( skl_preds[i] < 0.5 and bc_df_target[i] < 0.5):
            # print(f'{bcolors.OKGREEN}{bc_np[i][0]} {predictions[i]}{bcolors.ENDC}')
            correct_sklearn_count += 1
    print(100*correct_sklearn_count/bc_np.shape[0])



if __name__ == "__main__":
    main()
