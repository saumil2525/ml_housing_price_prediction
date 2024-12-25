import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter("ignore")

def impute_values(df, num_fillna="mean"):
    df_nulls = pd.DataFrame(df.isnull().sum(), columns=["count_val"])
    for col in df_nulls[df_nulls.count_val > 0].index.to_list():
        if df[col].dtype != "object":
            if num_fillna == "mean":
                # fill with median
                df[col] = df[col].fillna(df[col].mean())
            elif num_fillna == "median":
                df[col] = df[col].fillna(df[col].median())
            else:
                print("\nnum_fillna value must be 'mean' or 'median'")
        else:
            # most recurring values
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    print(f"\nImputed numerical missing values with {num_fillna} and categorical missing values with 'most frequent value'")

    return df

def feature_types(df):
    cat_features = [col for col in df.columns if df[col].dtype == "object"]
    num_features = [col for col in df.columns if not df[col].dtype == "object"]
    print(
        f"\nCatagorical feature count: {len(cat_features)} \nNumerical feature count: {len(num_features)}"
    )
    print(
        f"\nCatagorical features: \n\t{cat_features} \n\nNumerical features: \n\t{num_features}"
    )
    return cat_features, num_features

def one_hot_encoding(df, cat_features):
    from sklearn.preprocessing import OneHotEncoder
    # Initialize
    encoder = OneHotEncoder(sparse_output=False)
    # Fit and transform
    one_hot_encoded = encoder.fit_transform(df[cat_features])
    # df with the encoded columns
    one_hot_df = pd.DataFrame(
        one_hot_encoded, columns=encoder.get_feature_names_out(cat_features)
    )
    # concat the one-hot encoded columns with the original df
    df_encoded = pd.concat([df.drop(cat_features, axis=1), one_hot_df], axis=1)
    print(f"\nShape of the df after impute: {df_encoded.shape}")
    return df_encoded

def corr_with_target(df, threshold_corr=0.5, target_col="prices"):
    df_corr = pd.DataFrame(
        data=df.drop(target_col, axis=1).corrwith(df[target_col])
    ).reset_index()
    df_corr.rename(columns={"index": "col_name", 0: "corr_val"}, inplace=True)
    features_threshold_corr = df_corr[
        df_corr.corr_val.abs() > threshold_corr
    ].col_name.to_list()
    df_final = pd.concat([df[features_threshold_corr], df[target_col]], axis=1)
    return df_final, features_threshold_corr

def plot_corr_with_target(df, target_col):
    for ind, col in enumerate(df.drop(target_col, axis=1).columns):
        plt_title = f"{col} vs {target_col}"
        plt_name = f"plots/{ind+1}.{col}_vs_{target_col}.png"
        plt.xlabel(col)
        plt.ylabel(target_col)
        plt.title(plt_title)
        plt.scatter(df[col], df[target_col])
        plt.savefig(plt_name)

def write_df_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    with open("eda/01_df_info.txt", "w", encoding="utf-8") as f:
        f.write(s)
    print("\ndf info written to 'eda' directory")

def plot_correlation(df):
    # heatmap
    plt_name = f"plots/heatmap.png"
    fig = plt.gcf()  # Get the current figure
    fig.set_size_inches(18, 8)  # Set the size in inches
    sns.heatmap(df.corr(), annot=True)
    plt.savefig(plt_name)
    # pair plot
    plt_name = f"plots/pairplot.png"
    fig = plt.gcf()  # Get the current figure
    fig.set_size_inches(10, 10)  # Set the size in inches
    sns.pairplot(df)
    plt.savefig(plt_name)
    # plt.show()

def get_null_cols(df):
    df_nulls = pd.DataFrame(df.isnull().sum(), columns=['count_val'])
    df_nulls = df_nulls[df_nulls.count_val > 0]
    if df_nulls.shape[0] > 0:
        print(f"\nNumber of columns with nulls: {df_nulls[df_nulls.count_val > 0].shape[0]}")
    else:
        print("\nNo missing values")

def describe_features(df, cat_features, num_features):
    file_path = "eda/02_describe_categorical_feature.csv"
    df[cat_features].describe().T.to_csv(file_path)
    file_path = "eda/03_describe_numerical_feature.csv"
    df[num_features].describe().T.to_csv(file_path)
    print("\nFeatures described and written to 'eda' directory")
