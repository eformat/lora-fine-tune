import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

device = "cuda:0"
model_id = "ibm-granite/granite-3.0-8b-base"
model = AutoModelForCausalLM.from_pretrained(model_id, return_dict=True, load_in_8bit=True, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="cuda")

# Load the Lora model
batch = tokenizer(sys.argv[1:], return_tensors='pt').to(device)

with torch.cuda.amp.autocast():
  output_tokens = model.generate(**batch, max_new_tokens=50)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
