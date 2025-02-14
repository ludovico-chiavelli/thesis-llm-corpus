import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer (adjust to your specific LLaMA variant)
model_name = "gpt2"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def generate_text(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Read batch of prompts from a file
with open("prompts.txt", "r") as f:
    prompts = [line.strip() for line in f.readlines()]

results = [generate_text(prompt) for prompt in prompts]

# Save results
with open("generated_texts.txt", "w") as f:
    for prompt, result in zip(prompts, results):
        f.write(f"Prompt: {prompt}\nGenerated: {result}\n\n")

print("Text generation complete.")