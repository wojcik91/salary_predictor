import pandas as pd
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


if __name__ == "__main__":
    salary_df = pd.read_csv('salary.csv', na_values=['', ' '])
    (salary_data, salary_pred) = sort_input_data(salary_df)

    salary_X_train = pd.DataFrame(np.reshape(salary_data['workedYears'].values, (-1,1)))
    salary_X_pred = pd.DataFrame(np.reshape(salary_pred['workedYears'].values, (-1,1)))

    salary_Y_train = salary_data['salaryBrutto']
