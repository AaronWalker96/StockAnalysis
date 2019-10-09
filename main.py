import pandas as pd
from pandas_datareader import data
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression


def main():
    # We would like all available data from 01/01/2000 until 12/31/2016.
    start_date = '2015-01-01'
    end_date = '2019-10-09'

    # User pandas_reader.data.DataReader to load the desired data and add new columns with additional information
    df = data.DataReader('INPX', 'yahoo', start_date, end_date)
    df['percentage_h_l'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    df['percentage_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
    df = df[['Close', 'percentage_h_l', 'percentage_change', 'Volume']]
    print(df.tail())

    forecast_col = 'Close'
    df.fillna(value=-99999, inplace=True)
    forecast_out = int(math.ceil(0.01 * len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out)
    df.dropna(inplace=True)

    X = np.array(df.drop(['label'], 1))
    y = np.array(df['label'])
    X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)

    print(f'The confidence level is: {confidence}')


if __name__ == '__main__':
    main()