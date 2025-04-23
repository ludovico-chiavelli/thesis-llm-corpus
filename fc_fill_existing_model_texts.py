from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Concatenate all llama partials
llama_partial_dataframes = []
for partial in Path("llama-partials").glob("*.csv"):
    llama_partial_dataframes.append(pd.read_csv(partial))
llama_full = pd.concat(llama_partial_dataframes)

# Concatenate all gemma partials
gemma_partial_dataframes = []
for partial in Path("gemma-partials").glob("*.csv"):
    gemma_partial_dataframes.append(pd.read_csv(partial))
gemma_full = pd.concat(gemma_partial_dataframes)

# Concatenate all mistral partials
mistral_partial_dataframes = []
for partial in Path("mistral-partials").glob("*.csv"):
    mistral_partial_dataframes.append(pd.read_csv(partial))
mistral_full = pd.concat(mistral_partial_dataframes)

# Filter all dataframes from rows with no text in LLM_TEXT_1, LLM_TEXT_2, LLM_TEXT_3
llama_full = llama_full[llama_full["LLM_TEXT_1"].notna() & llama_full["LLM_TEXT_2"].notna() & llama_full["LLM_TEXT_3"].notna()]
gemma_full = gemma_full[gemma_full["LLM_TEXT_1"].notna() & gemma_full["LLM_TEXT_2"].notna() & gemma_full["LLM_TEXT_3"].notna()]
mistral_full = mistral_full[mistral_full["LLM_TEXT_1"].notna() & mistral_full["LLM_TEXT_2"].notna() & mistral_full["LLM_TEXT_3"].notna()]

# Add llama all partial texts per topic to full corpus if the topic exists in the full corpus. 
# There's only three texts per topic, you can add up to the amount of human texts that exist already for that topic
corpus_df = pd.read_csv("/home/nuvolari/GitHub/thesis-llm-corpus/final_corpus_unfiltered_hum_texts.csv", dtype=str, na_values=[""], keep_default_na=False)

def add_model_texts_to_full_corpus_entries(model_full_df, model_name):
    # For EFCAMDAT entries, there are min 1 and at most 10 human texts and for each topic we have 3 generated texts
    # So we can add min 1 and at most 3 generated texts per topic

    for index, model_row in tqdm(model_full_df.iterrows(), total=model_full_df.shape[0]):
        # Locate the topic in the corpus
        topic = model_row["TOPIC"]
        corpus_rows = corpus_df[corpus_df["TOPIC"] == model_row["TOPIC"]]
        model_texts_stack = [model_row["LLM_TEXT_1"], model_row["LLM_TEXT_2"], model_row["LLM_TEXT_3"]]
        if not corpus_rows.empty:
            for j, corpus_row in corpus_rows.iterrows():
                corpus_df.iloc[j, corpus_df.columns.get_loc(f"{model_name}_TEXT")] = model_texts_stack.pop(0)
                if not model_texts_stack:
                    break
    
            

# Call for each model
add_model_texts_to_full_corpus_entries(llama_full, "LLAMA")
add_model_texts_to_full_corpus_entries(gemma_full, "GEMMA")
add_model_texts_to_full_corpus_entries(mistral_full, "MISTRAL")

# Save the full corpus with the generated texts
corpus_df.to_csv("final_corpus_unf_texts_with_partials.csv", index=False)