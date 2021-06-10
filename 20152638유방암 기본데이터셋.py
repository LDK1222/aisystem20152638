import numpy as np
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

type(cancer)

dir(cancer)

cancer.data.shape
cancer.feature_names
cancer.target_names
cancer.target
np.bincount(cancer.target)

print(cancer.DESCR)
