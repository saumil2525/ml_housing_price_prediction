import pandas as pd
import warnings
warnings.simplefilter('ignore')
from sklearn.datasets import fetch_openml

# load dataset
housing = fetch_openml(name="house_prices", as_frame=True)
df_housing = pd.DataFrame(data=housing.data, columns=housing.feature_names)

# add target
df_housing['SalePrice'] = housing.target

# save dataset
df_housing.to_csv("data/ame_housing_data.csv", index=None)

# save description
with open("data/dataset_description.txt", "w") as f:
    f.write(housing.DESCR)

print("Dataset loading completed.")