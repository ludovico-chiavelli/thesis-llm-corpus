import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Initialize distributed
rank = int(os.environ["RANK"])
device = torch.device(f"cuda:{rank}")
torch.distributed.init_process_group("nccl", device_id=device)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    tp_plan="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_text(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
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