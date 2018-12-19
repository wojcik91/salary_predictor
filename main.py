import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np


def sort_input_data(df):
    data = pd.DataFrame()
    pred = pd.DataFrame()
    for index, row in df.iterrows():
        if pd.isnull(row[1]):
            pred = pred.append(row)
        else:
            data = data.append(row)

    return (data, pred)


def print_salary_predictions(years, salary):
    print('##### PREDICTED SALARY #####')
    print('workedYears\tsalaryBrutto')
    for prediction in zip(years, salary):
        print(f'{prediction[0]}\t|\t{prediction[1]}')


if __name__ == "__main__":
    salary_df = pd.read_csv('salary.csv', na_values=['', ' '])
    (salary_data, salary_pred) = sort_input_data(salary_df)

    salary_X_train = pd.DataFrame(np.reshape(salary_data['workedYears'].values, (-1,1)))
    salary_X_pred = pd.DataFrame(np.reshape(salary_pred['workedYears'].values, (-1,1)))

    salary_Y_train = salary_data['salaryBrutto']

    regr = linear_model.LinearRegression()

    regr.fit(salary_X_train, salary_Y_train)

    salary_Y_pred = regr.predict(salary_X_pred)

    print_salary_predictions(np.reshape(salary_X_pred.values, (1,-1))[0], salary_Y_pred)

    plt.scatter(salary_X_train, salary_Y_train, color='black')
    plt.plot(salary_X_pred, salary_Y_pred, color='blue', linewidth=3)

    plt.show()
