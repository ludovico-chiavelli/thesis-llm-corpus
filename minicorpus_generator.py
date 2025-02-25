import pandas as pd
from transformers import pipeline
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate text using LLM for minicorpus")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to the output CSV file")
    args = parser.parse_args()

    df = pd.read_csv("human_llm_corpus.csv", dtype=str, na_values=[""], keep_default_na=False)

    generator = pipeline('text-generation', model=args.model_name, tokenizer=args.model_name, device_map="auto", torch_dtype="auto", model_kwargs={"load_in_4bit": True})

    print(f"Generating text using LLM {args.model_name} for the first 10 rows of the minicorpus")

    for index, row in df.head(2).iterrows():
        prompt = row["PROMPT"]
        response = generator(prompt, max_new_tokens=100, num_return_sequences=1)
        df.loc[index, "LLM_NAME"] = args.model_name
        df.loc[index, "LLM_GENERATED_TEXT"] = response[0]['generated_text'][len(prompt):] # Only keep the generated text, not the prompt
    
    # Save dataframe to a new CSV file
    df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()