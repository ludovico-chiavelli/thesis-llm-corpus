# This script combines the model partials and puts them into a csv file

import pandas as pd
from pathlib import Path

def combine_model_partials():
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

def combine_into_final():
    model_partials = ("combined_llama_partials.csv", "combined_gemma_partials.csv", "combined_mistral_partials.csv")
    final_df = pd.DataFrame(columns=["CORPUS", "TOPIC_ID", "TOPIC", "HUMAN_TEXT", "PROMPT", "MODEL_NAME", "MODEL_TEXT"])

    mn_to_fn = {
        "llama": "meta-llama/Llama-3.1-8B-Instruct",
        "gemma": "google/gemma-2-9b-it",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3"
    }

    # load all models into DFs and remove other model columns.
    llama_df = pd.read_csv(model_partials[0])
    llama_df = llama_df.drop('GEMMA_TEXT', axis=1)
    llama_df = llama_df.drop('MISTRAL_TEXT', axis=1)

    gemma_df = pd.read_csv(model_partials[1])
    gemma_df = gemma_df.drop('LLAMA_TEXT', axis=1)
    gemma_df = gemma_df.drop('MISTRAL_TEXT', axis=1)

    mistral_df = pd.read_csv(model_partials[2])
    mistral_df = mistral_df.drop('LLAMA_TEXT', axis=1)
    mistral_df = mistral_df.drop('GEMMA_TEXT', axis=1)

    model_dfs = [(llama_df, "llama"), (gemma_df, "gemma"), (mistral_df, "mistral")]

    for df, model_name in model_dfs:
        for i, row in df.iterrows():
            new_row = {
                "CORPUS": row["CORPUS"],
                "TOPIC_ID": row["TOPIC_ID"],
                "TOPIC": row["TOPIC"],
                "HUMAN_TEXT": row["HUMAN_TEXT"],
                "PROMPT": row["PROMPT"],
                "MODEL_NAME": mn_to_fn[model_name],
                "MODEL_TEXT": row[f"{model_name.upper()}_TEXT"]
            }
            row_df = pd.DataFrame([new_row])
            final_df = pd.concat([final_df, row_df], axis=0, ignore_index=True)
    
    # Filter out rows with empty MODEL_TEXT
    final_df.dropna(subset=['MODEL_TEXT'])

    # Save the final dataframe to a csv file
    final_df.to_csv("final_corpus_uncleaned_unfiltered.csv", index=False)

if __name__ == "__main__":
    combine_into_final()
