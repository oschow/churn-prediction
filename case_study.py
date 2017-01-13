import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split


def load_scrub_churn():
    df = pd.read_csv('data/churn.csv',
                     parse_dates=['last_trip_date', 'signup_date'])
    dummies = pd.get_dummies(df['city'], drop_first=1)
    df['iphone'] = df['phone'].map({'Android': 0, 'iPhone': 1})
    df = pd.concat([df, dummies], axis=1)
    df['iphone'] = df['phone'].map({'Android': 0, 'iPhone': 1})
    churn_date = pd.to_datetime(['2014-07-01']) - pd.DateOffset(30)
    df['churn'] = df['last_trip_date'].apply(
        lambda x: x < churn_date).astype(bool)
    df.drop(['city', 'phone', 'last_trip_date'], inplace=True, axis=1)
    df = df[~pd.isnull(df['iphone'])]
    df['avg_rating_by_driver'].fillna(method='bfill', inplace=True)
    df['avg_rating_of_driver'].fillna(method='ffill', inplace=True)
    df['signup_date'] = df['signup_date'].apply(lambda x: x.toordinal())
    return df


def split_df(df):
    y = df.pop('churn').values
    X = df[['trips_in_first_30_days', 'luxury_car_user',
            'iphone', "King's Landing", 'Winterfell']].values
    return train_test_split(X, y, test_size=0.3, random_state=42)


def main():
    df = load_scrub_churn()
    X_train, X_test, y_train, y_test = split_df(df)

if __name__ == '__main__':
    main()
