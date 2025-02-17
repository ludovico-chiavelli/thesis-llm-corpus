import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained model and tokenizer
model_name = 'openai-community/gpt2'
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def generate_text(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Read batch of prompts from a file
with open("../prompts.txt", "r") as f:
    prompts = [line.strip() for line in f.readlines()]

results = [generate_text(prompt) for prompt in prompts]

# Write to CSV file
with open('gpt2_output.csv', 'w', newline='') as csvfile:
    fieldnames = ['LLM Used', 'Prompt', 'Response']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for prompt, generated_text in zip(prompts, results):
        writer.writerow({'Prompt': prompt, 'LLM Used': model_name, 'Response': generated_text})

print("Text generation complete. Output written to gpt2_output.csv")