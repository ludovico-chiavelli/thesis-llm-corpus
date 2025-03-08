import pandas as pd
from transformers import pipeline
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Generate text using LLM for minicorpus")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--start_line", type=int, required=True, help="Index of the first line to generate text for")
    parser.add_argument("--end_line", type=int, required=True, help="Index of the last line to generate text for")
    # this is optional. So you can run it with other models, who don't have a folder and filename specified
    parser.add_argument("--output_csv", type=str, required=False, help="Path to the output CSV file")
    args = parser.parse_args()

    df = pd.read_csv("final_corpus_unfiltered_texts.csv", dtype=str, na_values=[""], keep_default_na=False)

    generator = pipeline(
        'text-generation',
        model=args.model_name,
        tokenizer=args.model_name,
        device_map="auto",
        torch_dtype="auto",
        model_kwargs={"offload_folder": "offload"}
    )

    print(f"Generating text using LLM {args.model_name} for lines {args.start_line} to {args.end_line} of the corpus scaffolding")

    for index, row in df.iloc[args.start_line:args.end_line].iterrows():
        prompt = row["PROMPT"]

        # Model name to column name mapping
        mn_to_cn = {
            "meta-llama/Llama-3.1-8B-Instruct": "LLAMA_TEXT",
            "google/gemma-2-9b-it": "GEMMA_TEXT",
            "mistralai/Mistral-7B-Instruct-v0.3": "MISTRAL_TEXT"
        }

        # Check if dataframe row already has generated text for the model
        column_name = mn_to_cn[args.model_name]
        if row[column_name] != "":
            continue  # Skip this row if it already has generated text
        else:
            response = generator(prompt, max_new_tokens=200, num_return_sequences=1, do_sample=True, top_k=50, temperature=1)
            df.loc[index, column_name] = response[0]['generated_text'][len(prompt):] # Only keep the generated text, not the prompt
    
    full_corpus_dir_prefix = "fc-"
    if args.model_name == "meta-llama/Llama-3.1-8B-Instruct":
        dir_name = f"{full_corpus_dir_prefix}llama-partials"
        file_name = f"fc_llama_partial_{args.start_line}_{args.end_line}.csv"
        # Save dataframe to a new CSV file
        df.to_csv(f"{dir_name}/{file_name}", index=False)
    elif args.model_name == "google/gemma-2-9b-it":
        dir_name = f"{full_corpus_dir_prefix}gemma-partials"
        file_name = f"fc_gemma_partial_{args.start_line}_{args.end_line}.csv"
        # Save dataframe to a new CSV file
        df.to_csv(f"{dir_name}/{file_name}", index=False)
    elif args.model_name == "mistralai/Mistral-7B-Instruct-v0.3":
        dir_name = f"{full_corpus_dir_prefix}mistral-partials"
        file_name = f"fc_mistral_partial_{args.start_line}_{args.end_line}.csv"
        # Save dataframe to a new CSV file
        df.to_csv(f"{dir_name}/{file_name}", index=False)
    elif args.model_name not in ["meta-llama/Llama-3.1-8B-Instruct", "google/gemma-2-9b-it", "mistralai/Mistral-7B-Instruct-v0.3"] and args.output_csv == None:
        print("Error: You must provide an output CSV file for models other than Llama, Gemma, and Mistral.")
        sys.exit(1)
    elif args.output_csv:
        df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()