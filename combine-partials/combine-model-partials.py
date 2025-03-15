# This script combines the model partials and puts them into a csv file

import pandas as pd
from pathlib import Path

# Select all files from fc-[model_name]-partials folder
model_name = "gemma"
partials_folder = Path(f"../fc-{model_name}-partials")
files = list(partials_folder.glob("*.csv"))

# Create a new dataframe to store the combined partials. Use the same column names
merged_df = pd.DataFrame(columns=["CORPUS", "TOPIC_ID", "TOPIC", "HUMAN_TEXT", "PROMPT", "LLAMA_TEXT", "GEMMA_TEXT", "MISTRAL_TEXT"])

for file in files:
    partial_df = pd.read_csv(file)
    # Add the rows specified in the filename to the merged_df
    start_line, end_line = file.stem.split("_")[-2:]
    start_line = int(start_line)
    end_line = int(end_line)
    merged_df = pd.concat([merged_df, partial_df.loc[start_line:end_line]])

# Save the combined dataframe to a csv file
output_file = f"combined_{model_name}_partials.csv"
merged_df.to_csv(output_file, index=False)
print(f"Combined partials saved to {output_file}")
