import pandas as pd
from datasets import load_dataset
from datasets import Dataset, DatasetDict

rd_ds = load_dataset("badrex/llm-emoji-dataset")

# Convert to pandas dataframe for convenient processing
rd_df = pd.DataFrame(rd_ds['train'])

# Combine the two attributes into an instruction string
rd_df['instruction'] = 'The emjoi character: '+ rd_df['unicode']+' has the description: '+ rd_df['LLM description']

# Drop
rd_df.drop(columns=['character', 'unicode', 'LLM description', 'short description', 'tags'], inplace=True)

# random
rd_df_sample = rd_df.sample(n=5, random_state=42)
print(rd_df_sample.to_string())

# Create a dataset
ds = DatasetDict()
tds = Dataset.from_pandas(rd_df)
ds['train'] = tds

ds.push_to_hub("eformat/emoji-lora-train")
