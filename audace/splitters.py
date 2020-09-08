import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# Utility function to reshape features of any shape from a pandas Serie
# into a 2D array with shape (n_samples, product(feature dimensions))
def serie_to_2D(s):
    x = np.stack(s)
    return x.reshape(x.shape[0], -1)


def splitTrainTestStratified(main_df,
                             train_size,
                             feature_name,
                             label_name,
                             key_name
                             ):

    key_values = main_df[key_name].unique()

    df_train = None
    df_test = None
    for key_value in key_values:
        df = main_df.loc[main_df[key_name] == key_value]
        tdf_train, tdf_test = train_test_split(df, train_size=train_size)
        df_train = pd.concat([df_train, tdf_train], ignore_index=True)
        df_test = pd.concat([df_test, tdf_test], ignore_index=True)

    return (serie_to_2D(df_train[feature_name]),
            serie_to_2D(df_test[feature_name]),
            serie_to_2D(df_train[label_name]),
            serie_to_2D(df_test[label_name]))


def balanceDownSampleMajority(df, axis):
    label_values = df[axis].unique()

    min = 999999999
    min_label_value = -1
    for i, label_value in enumerate(label_values):
        dfv = df[df[axis] == label_value]
        card = len(dfv.index)
        if card < min:
            min = card
            min_label_value = label_value

    df_resampled = df[df[axis] == min_label_value]
    for i, label_value in enumerate(label_values):
        if (label_value != min_label_value):
            dfv = df[df[axis] == label_value]
            df_aux_resampled = resample(
                dfv,
                replace=False,    # sample without replacement
                n_samples=min,    # to match minority class
            )

            df_resampled = pd.concat([df_resampled, df_aux_resampled])

    return df_resampled


def balanceUpSampleMinority(df, axis):
    label_values = df[axis].unique()

    max = 0
    max_label_value = -1
    for i, label_value in enumerate(label_values):
        dfv = df[df[axis] == label_value]
        card = len(dfv.index)
        if card > max:
            max = card
            max_label_value = label_value

    df_resampled = df[df[axis] == max_label_value]
    for i, label_value in enumerate(label_values):
        if (label_value != max_label_value):
            dfv = df[df[axis] == label_value]
            df_aux_resampled = resample(
                dfv,
                replace=True,     # sample with replacement
                n_samples=max,    # to match majority class
            )

            df_resampled = pd.concat([df_resampled, df_aux_resampled])

    return df_resampled


def splitTrainTestFold(main_df,
                       feature_name,
                       label_name,
                       fold_name,
                       fold_value,
                       balance_strategy=0
                       ):

    df_train = main_df.loc[main_df[fold_name] != fold_value]
    df_test = main_df.loc[main_df[fold_name] == fold_value]

    if balance_strategy == -1:
        df_train = balanceDownSampleMajority(df_train, label_name)
        df_test = balanceDownSampleMajority(df_test, label_name)
    elif balance_strategy == 1:
        df_train = balanceUpSampleMinority(df_train, label_name)
        df_test = balanceUpSampleMinority(df_test, label_name)

    return (serie_to_2D(df_train[feature_name]),
            serie_to_2D(df_test[feature_name]),
            serie_to_2D(df_train[label_name]),
            serie_to_2D(df_test[label_name]))


def splitTrainValidTestFold(main_df,
                            feature_name,
                            label_name,
                            fold_name,
                            fold_value,
                            valid_frac
                            ):
    df_train_valid = main_df.loc[main_df[fold_name] != fold_value]
    df_test = main_df.loc[main_df[fold_name] == fold_value]
    df_train, df_valid = train_test_split(df_train_valid, test_size=valid_frac)

    return (serie_to_2D(df_train[feature_name]),
            serie_to_2D(df_valid[feature_name]),
            serie_to_2D(df_test[feature_name]),
            serie_to_2D(df_train[label_name]),
            serie_to_2D(df_valid[label_name]),
            serie_to_2D(df_test[label_name]))


def splitTrainTest(df,
                   train_size,
                   feature_name,
                   label_name
                   ):
    df_train, df_test = train_test_split(df, train_size=train_size)

    return (serie_to_2D(df_train[feature_name]),
            serie_to_2D(df_test[feature_name]),
            serie_to_2D(df_train[label_name]),
            serie_to_2D(df_test[label_name]))


def splitTrainValidTest(df,
                        rvalid,
                        rtest,
                        feature_name,
                        label_name
                        ):

    # First split into training+validation and test subsets
    df_train_val, df_test = train_test_split(df, test_size=rtest)
    # then split training+validation into training and validation
    df_train, df_valid = train_test_split(
        df_train_val, test_size=rvalid/(1.0-rtest))

    return (serie_to_2D(df_train[feature_name]),
            serie_to_2D(df_valid[feature_name]),
            serie_to_2D(df_test[feature_name]),
            serie_to_2D(df_train[label_name]),
            serie_to_2D(df_valid[label_name]),
            serie_to_2D(df_test[label_name]))
