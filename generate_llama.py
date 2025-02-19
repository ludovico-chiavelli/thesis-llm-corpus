from transformers import pipeline


model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

generator = pipeline('text-generation', model=model_name, tokenizer=model_name, device_map="auto", torch_dtype="auto")

print(generator("Tell me a story about a dragon", max_length=100, num_return_sequences=3))