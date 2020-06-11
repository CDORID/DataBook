import pandas as pd
import numpy as np

def random_undersample(X_train,Y_train, target_var = str, minor_class = 1, limit_sample = 200):
    """
    Simple undersampling, test keep the same proportion

    """
    # Rebuild training set
    training_set = pd.concat([X_train,Y_train],axis = 1)

    # Choose major class, and minor class as 1 and 0s
    major_class = 1-minor_class
    print("Previous split : \n0 : ",training_set[training_set[target_var] == 0].shape[0],"\n1 : ",
          training_set[training_set[target_var] == 1].shape[0])


    #Define which values will be removed randomly
    major_indices  = training_set[training_set[target_var] == major_class].index
    minor_indices = training_set[training_set[target_var] == minor_class].index
    random_indices = np.random.choice(major_indices, limit_sample, replace=False)
    kept_indices = np.sort(np.append(minor_indices,random_indices))


    #Redefine the training set according to the undersampling
    training_set_undersampled  = training_set.loc[kept_indices]


    print("New split : \n0 : ",training_set_undersampled[training_set_undersampled[target_var] == 0].shape[0],"\n1 : ",
          training_set_undersampled[training_set_undersampled[target_var] == 1].shape[0])


    training_set_undersampled
    X_train_under = training_set_undersampled.drop(target_var, axis =1)
    Y_train_under = training_set_undersampled[target_var]

    return X_train_under, Y_train_under
