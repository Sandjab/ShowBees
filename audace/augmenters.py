from tqdm.auto import tqdm


# this is probably a very badly written code from an optimisation standpoint
def getScratchedFeatureRows(param_df, feature_name, value=None):
    # create empty result dataframe with same structure as input df
    result_df = param_df.iloc[0:0, :].copy()

    # walk dataframe rows
    for index, row in tqdm(param_df.iterrows(),
                           desc="Augmenting",
                           mininterval=0.5,
                           total=len(param_df.index)):
        # if no value was specified, use mean as scratching value
        if value is None:
            value = row[feature_name].mean()

        # walk feature to generate a scratched sample per feature row
        # at this point i'm unsure about true copy vs references
        # so better safe than sorry
        rng = range(row[feature_name].shape[0])
        for i in rng:
            cp_row = row.copy()
            feature = cp_row[feature_name].copy()
            feature[i] = value
            cp_row.at[feature_name] = feature
            result_df = result_df.append(cp_row, ignore_index=True)

    return result_df


def addScratchedFeatureRows(param_df, feature_name, value=None):
    param_df = param_df.append(
        getScratchedFeatureRows(param_df, feature_name, value),
        ignore_index=True)

    return param_df
