import argparse
import sys
from transformers import pipeline

def main():
    parser = argparse.ArgumentParser(description="Generate text using LLM")
    parser.add_argument("--prompt", type=str, required=False, help="Prompt to generate text from")
    parser.add_argument("--max_length", type=int, required=False, help="Maximum length of the generated text")
    parser.add_argument("--num_return_sequences", type=int, required=False, default=1, help="Number of sequences to use")
    parser.add_argument("--prompts_file", type=str, required=False, help="Path to the file containing prompts")
    parser.add_argument("--output_file", type=str, required=False, help="Path to the output file")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    args = parser.parse_args()

    generator = pipeline('text-generation', model=args.model_name, tokenizer=args.model_name, device_map="auto", torch_dtype="auto")

    if not args.prompt and not (args.prompts_file and args.output_file):
        print("Error: You must provide either a prompt or both a prompts file and an output file.")
        sys.exit(1)

    if args.prompts_file and args.output_file:
        with open(args.prompts_file, "r") as f:
            prompts = f.readlines()
        
        with open(args.output_file, "w") as f:
            for prompt in prompts:
                f.write(generator(prompt, max_length=args.max_length, num_return_sequences=args.num_return_sequences))

    else:
        print(generator(args.prompt, max_length=args.max_length, num_return_sequences=args.num_return_sequences))




if __name__ == "__main__":
    main()
    