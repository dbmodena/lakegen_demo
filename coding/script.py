import pandas as pd

file_path = '/home/gabriele.martinelli/LakeGen/LakeGen_T/Data/data_csv/avocado.csv'
df = pd.read_csv(file_path)

filtered_df = df[
    (df['year'] == 2017) &
    (df['type'].str.lower() == 'organic') &
    (df['region'].str.lower() == 'chicago')
]

if filtered_df.empty:
    print("ERROR_EMPTY: No matching records found for those filters")
else:
    avg_cost = filtered_df['AveragePrice'].mean()
    if pd.isna(avg_cost):
        print("ERROR_EMPTY: No matching records found for those filters")
    else:
        print(f"The average cost of organic avocados sold in Chicago in 2017 was: {avg_cost}")