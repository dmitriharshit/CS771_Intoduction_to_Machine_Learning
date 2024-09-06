import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

def product_matrix(arr):
     # Compute the matrix product (multiplication) of the array with itself
    kr_product = khatri_rao(arr[:, None], arr[:, None]).reshape(len(arr), len(arr))
    product_pairs = np.triu(kr_product, k=1)
    return product_pairs[product_pairs != 0]

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

    X_new_train = my_map(X_train)
    model=LogisticRegression()
    model.fit(X_new_train, y_train)
    w = model.coef_.reshape((528,))
    b = model.intercept_
    return w,b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
    d = 1-2*X
    x = np.fliplr(np.cumprod(np.fliplr(d), axis=1))
    X_new = np.apply_along_axis(product_matrix, axis=1, arr=x)
    X_new = np.concatenate((X_new, x), axis=1)
    return X_new



