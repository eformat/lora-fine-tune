from huggingface_hub import HfApi
api = HfApi()

model_id = "eformat/granite-3.0-8b-instruct-Q4_K_M-GGUF"
api.create_repo(model_id, exist_ok=True, repo_type="model")
api.upload_file(
    path_or_fileobj="/home/ec2-user/lora-fine-tune/models/granite-3.0-8b-instruct/granite-3.0-8b-instruct-Q4_K_M.gguf",
    path_in_repo="granite-3.0-8b-instruct-Q4_K_M.gguf",
    repo_id=model_id,
)
