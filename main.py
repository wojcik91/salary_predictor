import pandas as pd

def sort_input_data(df):
    data = []
    pred = []
    for index, row in df.iterrows():
        if pd.isnull(row[1]):
            pred.append(row)
        else:
            data.append(row)

    return (data, pred)
        

if __name__ == "__main__":
    salary_df = pd.read_csv('salary.csv', na_values=['', ' '])
    print(salary_df)
    (salary_data, salary_pred) = sort_input_data(salary_df)