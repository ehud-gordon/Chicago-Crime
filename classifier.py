import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import time

def process_data(df):
    # drop columns
    cols_to_keep = ['Date', 'Year', 'Arrest', 'Domestic', 'Beat',
                    'District', 'Ward', 'Community Area']
    df = df[cols_to_keep].copy()
    # Extract Time
    date = pd.to_datetime(df.Date, format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    df.drop(columns=['Date'], inplace=True)
    df['month'] = date.dt.month
    df['hour'] = date.dt.hour
    # fill NA values
    df.ffill(inplace=True), df.bfill(inplace=True)
    # optimize DataFrame
    downcast(df)
    return df

def downcast(df):
    df.District = df.District.astype('int64').astype('category')
    df.Ward = df.Ward.astype('int64').astype('category')
    df['Community Area'] = df['Community Area'].astype('int64').astype('category')
    df.Beat = df.Beat.astype('int64').astype('category')
    df_num = df.select_dtypes(include=['int', 'float'])
    converted = df_num.apply(pd.to_numeric, downcast='unsigned', errors='coerce')
    df[converted.columns] = converted

def classify(x_test):
    x_test = process_data(x_test)
    model = pickle.load(open('weights.txt', 'rb'))
    y_pred = model.predict(x_test)
    return y_pred

def train_model(x_train, y_train):
    x_train = process_data(x_train)
    RandomFor = RandomForestClassifier(n_estimators=70, min_samples_split=30, bootstrap=True, max_depth=50,
                                       min_samples_leaf=25)
    clf = RandomFor.fit(x_train, y_train)
    pickle.dump(clf, open('weights.txt', 'wb'))
