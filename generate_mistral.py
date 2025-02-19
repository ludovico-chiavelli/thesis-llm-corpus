import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

model_name = ''
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_text(prompt, max_length=200):
    model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**model_inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)