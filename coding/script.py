import pandas as pd

# Load the datasets
df_score = pd.read_csv('/home/gabriele.martinelli/LakeGen/LakeGen_T/Data/data_csv/2019.csv')
df_cost = pd.read_csv('/home/gabriele.martinelli/LakeGen/LakeGen_T/Data/data_csv/Cost_of_Living_Index_by_Country_2024.csv')

# Normalize country names for merging to handle case sensitivity
df_score['country_key'] = df_score['Country or region'].str.lower()
df_cost['country_key'] = df_cost['Country'].str.lower()

# Merge the datasets on the normalized country keys
merged_df = pd.merge(df_score, df_cost, left_on='country_key', right_on='country_key', how='inner')

# Check if the merge resulted in an empty dataframe
if merged_df.empty:
    print("ERROR_EMPTY: No matching records found for those filters")
else:
    # Find the row with the highest score
    max_score_row = merged_df.loc[merged_df['Score'].idxmax()]

    # Check if the Local Purchasing Power Index is NaN
    if pd.isna(max_score_row['Local Purchasing Power Index']):
        print("ERROR_EMPTY: No matching records found for those filters")
    else:
        print(f"Country: {max_score_row['Country or region']}, Local Purchasing Power Index: {max_score_row['Local Purchasing Power Index']}")