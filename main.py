import pandas as pd

if __name__ == "__main__":
    salary_df = pd.read_csv('salary.csv', na_values=['', ' '])
    print(salary_df)