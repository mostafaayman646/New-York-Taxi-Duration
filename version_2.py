import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge


def predict_eval(model, train, train_features, name):
    y_train_pred = model.predict(train[train_features])
    r2 = r2_score(train.log_trip_duration, y_train_pred)
    print(f"{name} R2 = {r2:.4f}")

def approach(train, test):
    numeric_features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude','distance']
    categorical_features = ['dayofweek', 'month', 'hour', 'dayofyear', 'passenger_count']
    train_features = categorical_features + numeric_features

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('scaling', StandardScaler(), numeric_features)
        ]
        , remainder = 'passthrough'
    )

    pipeline = Pipeline(steps=[
        ('ohe', column_transformer),
        ('regression', Ridge())
    ])

    model = pipeline.fit(train[train_features], train.log_trip_duration)
    predict_eval(model, train, train_features, "train")
    predict_eval(model, test, train_features, "test")

def correlation(df):
    df_corr = df
    corr = df_corr.corr()
    fig = plt.figure(figsize=(15,10))
    sns.heatmap(corr,annot=True,linewidths=.5,cmap='coolwarm',vmin=-1,vmax=1,center=0)
    plt.show()

def plot(df):
    columns = df.columns
    plt.figure(figsize=(5,5))
    sns.pairplot(df, x_vars=columns[0], y_vars=columns[1], height=4, aspect=1, kind='scatter')
    plt.show()

def remove_outliers(df, cols, k=2.1):
    print(f"before removing outliers: {df.shape}")
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    print(f"after removing outliers: {df.shape}")
    return df


def prepare_data(df):
    df.drop(columns=['id'], inplace=True)

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['dayofweek'] = df.pickup_datetime.dt.dayofweek
    df['month'] = df.pickup_datetime.dt.month
    df['hour'] = df.pickup_datetime.dt.hour
    df['dayofyear'] = df.pickup_datetime.dt.dayofyear

    df['log_trip_duration'] = np.log1p(df.trip_duration)

    # Vectorized haversine distance
    R = 6371
    lat1 = np.radians(df['pickup_latitude'])
    lon1 = np.radians(df['pickup_longitude'])
    lat2 = np.radians(df['dropoff_latitude'])
    lon2 = np.radians(df['dropoff_longitude'])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    df['distance'] = 2 * R * np.arcsin(np.sqrt(a))

    # correlation(train[['distance','log_trip_duration']]) #Distance has highest correlation

    # plot(df[['distance','log_trip_duration']]) #seems as square root function

    df['distance'] = np.sqrt(df['distance'])

    # put trip_duration at the end
    trip_duration_col = df.pop('trip_duration')
    df['trip_duration'] = trip_duration_col



if __name__ == '__main__':
    train = pd.read_csv('split/train.csv')
    test = pd.read_csv('split/val.csv')
    
    prepare_data(train)
    prepare_data(test)
    
    # Remove outliers from both sets
    train = remove_outliers(train, ['log_trip_duration', 'distance'])
    test = remove_outliers(test, ['log_trip_duration', 'distance'])
    
    approach(train, test)