import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Initialize distributed
os.environ["LOCAL_RANK"] = os.environ["SLURM_PROCID"]
rank = int(os.environ["RANK"])
device = torch.device(f"cuda:{rank}")
torch.distributed.init_process_group("nccl", device_id=device)

# Retrieve tensor parallel model
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.1-8B',                                                                        
    device_map = 'auto',                                                                      
    torch_dtype = torch.bfloat16,                                                             
    load_in_8bit = True,   
)

# Prepare input tokens
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = "Can I help"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Distributed run
outputs = model(inputs)