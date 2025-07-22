import pandas as pd

df = pd.read_csv('key_gt.csv')

df_presence = df[df['presence'] == '1']

# --- 2. Define grouping columns and perform sampling ---
grouping_cols = ['n_keys', 'type', 'difficulty']

# Group by the columns and sample 1 row from each group.
# A random_state is used for reproducibility. Remove it for different results each run.
sampled_df = df_presence.groupby(grouping_cols, as_index=False).sample(n=1, random_state=42)

# --- 3. Remove sampled rows from the original dataframe ---
remaining_df = df.drop(sampled_df.index)

# --- 4. Format the sampled data as a Markdown table ---
# We select a subset of columns for better readability in the output.
columns_to_display = ['nl', 'keywords']
markdown_table = sampled_df[columns_to_display].to_markdown(index=False)

# --- 5. Print the results ---
print("## Sampled Rows (One per n_keys/type/difficulty combination)")
print(markdown_table)
print("\n" + "="*80 + "\n")
print(f"Original DataFrame had {len(df)} rows.")
print(f"Sampled {len(sampled_df)} rows.")
print(f"The remaining DataFrame now has {len(remaining_df)} rows.")
