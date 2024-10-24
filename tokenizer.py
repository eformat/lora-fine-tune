import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "ibm-granite/granite-3.0-8b-base"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained(model_id, stop_sequence=[])
print(tokenizer)

print(tokenizer.vocab_size)

token_data = tokenizer.encode("tokenizer tokenizes text into tokens")
tokens = tokenizer.convert_ids_to_tokens(token_data)
print(tokens)

token_data = tokenizer.encode("😃")
tokens = tokenizer.convert_ids_to_tokens(token_data)
print(tokens)

print(tokenizer.get_vocab()["foo"])
