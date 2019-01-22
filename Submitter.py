import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')

# Get data from avaialble sources
train_df = pd.read_csv("data/train.csv", parse_dates=["first_active_month"])
target = train_df.target
test_df = pd.read_csv("data/test.csv", parse_dates=["first_active_month"])
# historical_df = pd.read_csv("data/historical_transactions.csv", parse_dates=["purchase_date"])
merchants_df = pd.read_csv("data/merchants.csv")
new_merchant_df = pd.read_csv("data/new_merchant_transactions.csv", parse_dates=["purchase_date"])

# Additional data constructed from history
spend_by_card_df = pd.read_csv("data/spend_by_card.csv")
city_by_card_df = pd.read_csv("data/city_by_card.csv")
state_by_card_df = pd.read_csv("data/state_by_card.csv")
merchant_by_card_df = pd.read_csv("data/merchant_by_card.csv")
merchant_cat_by_card_df = pd.read_csv("data/merchant_category_by_card.csv")
subsector_by_card_df = pd.read_csv("data/subsector_by_card.csv")
category_1_by_card_df = pd.read_csv("data/category_1_by_card.csv")
category_2_by_card_df = pd.read_csv("data/category_2_by_card.csv")
category_3_by_card_df = pd.read_csv("data/category_3_by_card.csv")

# add all of the new columns to train_df
train_df["year"] = train_df["first_active_month"].dt.year
train_df["month"] = train_df["first_active_month"].dt.month

train_df = pd.merge(train_df, spend_by_card_df, on='card_id')
train_df = pd.merge(train_df, city_by_card_df, on='card_id')
train_df = pd.merge(train_df, state_by_card_df, on='card_id')
train_df = pd.merge(train_df, merchant_by_card_df, on='card_id')
train_df = pd.merge(train_df, merchant_cat_by_card_df, on='card_id')
train_df = pd.merge(train_df, subsector_by_card_df, on='card_id')
train_df = pd.merge(train_df, category_1_by_card_df, on='card_id')
train_df = pd.merge(train_df, category_2_by_card_df, on='card_id')
train_df = pd.merge(train_df, category_3_by_card_df, on='card_id')

# Make categorical dummies of pretty much everything
train_df = pd.get_dummies(data=train_df,
                           columns=[
                               'year', 'month',
                               'feature_1', 'feature_2', 'feature_3',
                               'category_1', 'category_2', 'category_3',
                               'city_id', 'state_id',
                               'merchant_id', 'merchant_category_id',
                               'subsector_id'
                           ])


# add all of the new columns to test_df
test_df["year"] = test_df["first_active_month"].dt.year
test_df["month"] = test_df["first_active_month"].dt.month

test_df = pd.merge(test_df, spend_by_card_df, on='card_id')
test_df = pd.merge(test_df, city_by_card_df, on='card_id')
test_df = pd.merge(test_df, state_by_card_df, on='card_id')
test_df = pd.merge(test_df, merchant_by_card_df, on='card_id')
test_df = pd.merge(test_df, merchant_cat_by_card_df, on='card_id')
test_df = pd.merge(test_df, subsector_by_card_df, on='card_id')
test_df = pd.merge(test_df, category_1_by_card_df, on='card_id')
test_df = pd.merge(test_df, category_2_by_card_df, on='card_id')
test_df = pd.merge(test_df, category_3_by_card_df, on='card_id')

# Make categorical dummies of pretty much everything
test_df = pd.get_dummies(data=test_df, 
                           columns=[
                               'year', 'month',
                               'feature_1', 'feature_2', 'feature_3',
                               'category_1', 'category_2', 'category_3',
                               'city_id', 'state_id',
                               'merchant_id', 'merchant_category_id',
                               'subsector_id'
                           ])





# set data for regression
target = pd.DataFrame(train_df.target, columns=["target"])
X = train_df.drop(columns=['first_active_month', 'card_id', 'target'])
#X = train_df[["feature_1", "feature_2", "feature_3", "spend", "purchases"]]
y = target

lm = linear_model.LinearRegression()
model = lm.fit(X,y)

X_test = test_df.drop(columns=['first_active_month', 'card_id'])
result = test_df
result["target"] = lm.predict(X_test)

print(result[["card_id", "target"]])
result[["card_id", "target"]].to_csv('submission1.csv', header=["card_id", "target"], index=False)

lm.score(X,y)

print("Done")
