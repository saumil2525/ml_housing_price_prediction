import pandas as pd
from common_functions import (
    write_df_info,
    get_null_cols,
    feature_types,
    describe_features,
    impute_values,
    one_hot_encoding,
    corr_with_target,
    plot_correlation,
    plot_corr_with_target,
)
import matplotlib.pyplot as plt
import warnings

# load data
file_path = "./data/ame_housing_data.csv"
df_housing = pd.read_csv(file_path)

# print shape
print(f"\nRows: {df_housing.shape[0]}")
print(f"\nColumns: {df_housing.shape[1]}")

# print columns
print(f"\nColumns: {df_housing.columns}")

# writing info file
write_df_info(df_housing)

# get feature types
cat_features, num_features = feature_types(df=df_housing)

# describe features
describe_features(df_housing, cat_features, num_features)

# count null cols
get_null_cols(df_housing)

# impute nulls
df_housing = impute_values(df_housing)

# conducting one hot encoding
df_encoded = one_hot_encoding(df_housing, cat_features)

# get correlation with "SalePrice"
df_final, _ = corr_with_target(df_encoded, threshold_corr=0.5, target_col="SalePrice")

# plot correlation
plot_correlation(df_final)

# plot feature vs target
plot_corr_with_target(df_final, target_col="SalePrice")

# save final_df
df_final.to_csv("data/final.csv", index=None)
