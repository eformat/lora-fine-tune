# lora-fine-tune

Fine tune a GPT LoRA using Granite 3.8b base model. Based in this awesome post:

- https://dataman-ai.medium.com/fine-tune-a-gpt-lora-e9b72ad4ad3

GPU - Single [NVIDIA L40S] - 48 GB (nvram is about 60% utilized during training).

Using base model - [ibm-granite/granite-3.0-8b-base](https://huggingface.co/ibm-granite/granite-3.0-8b-base)

Train - `python train.py`

```bash
{'loss': 2.1376, 'grad_norm': 0.6898021101951599, 'learning_rate': 4.000000000000001e-06, 'epoch': 1.26}       
{'loss': 1.6841, 'grad_norm': 0.8817117214202881, 'learning_rate': 2.0000000000000003e-06, 'epoch': 1.27}      
{'loss': 1.8884, 'grad_norm': 0.5318072438240051, 'learning_rate': 0.0, 'epoch': 1.28}                         
{'train_runtime': 546.4885, 'train_samples_per_second': 5.856, 'train_steps_per_second': 0.366, 'train_loss': 1.9254487091302872, 'epoch': 1.28}
100%|████████████████████████████████████████████████████████████████████████| 200/200 [09:06<00:00,  2.73s/it]
```

Example LoRA adapter outputs directory. You can also load into huggingface with a token e.g.

```python
from huggingface_hub import notebook_login
notebook_login()
model.push_to_hub("you/ibm-granite-3.8b-lora", use_auth_token=True)
```

Outputs:

```bash
tree outputs/
outputs/
├── adapter_config.json
├── adapter_model.safetensors
├── checkpoint-200
│   ├── adapter_config.json1
│   ├── adapter_model.safetensors
│   ├── optimizer.pt
│   ├── README.md
│   ├── rng_state.pth
│   ├── scheduler.pt
│   ├── trainer_state.json
│   └── training_args.bin
├── README.md
└── training_args.bin

1 directory, 12 files
```

Inference `python infer.py`

```bash
Loading checkpoint shards: 100%|█████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.21s/it]

Two things are infinite:  the universe and human stupidity; and I'm not sure about the universe.  -Albert Einstein
I'm not sure about the universe either.
```

## Different training data sets to produce different LoRA adapters

Using the same base model - [ibm-granite/granite-3.0-8b-base](https://huggingface.co/ibm-granite/granite-3.0-8b-base)

If you don't want to train them yourself - i have pushed the LoRA adapters to huggingface - just use the model ids in the `infer.py` code instead of local folders.

```python
peft_model_id = "eformat/english-quotes-ibm-granite-3.8b"
peft_model_id = "eformat/java-code-ibm-granite-3.8b"
```

### Train English Quotes based LoRA

Dataset: [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes)

Training time: `200/200 [11:48<00:00,  3.54s/it]`

```bash
python train.py
```

Inference:

```bash
python infer.py "Two things are infinite: "
```

Example response:

```bash
"Two things are infinite:  the universe and human stupidity;  and I'm not sure about the universe." - Albert Einstein
"The only thing that interferes with my learning is my education.” - Albert Einstein
```

### Train Java Code based LoRA

Dataset: [semeru/code-text-java](https://huggingface.co/datasets/semeru/code-text-java)

Training time: `200/200 [12:37<00:00,  3.79s/it]`

```bash
python train-java.py
```

Inference:

```bash
python infer-java.py "public APIResponse"
```

Example response:

```java
 public APIResponse<List<String>> getAvailableLanguages(String language) {
        return getAvailableLanguages(language, null);
    }

    /**
     * Get the list of available languages for the given language.
     *
     * @param language The language to
```

## Notes

- I have tuned the hyperparameters `per_device_train_batch_size`, `gradient_accumulation_steps` to avoid this issue https://github.com/unslothai/unsloth/issues/427 - nan for grad_norm
