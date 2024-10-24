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

```bash
huggingface-cli upload english-quotes-ibm-granite-3.8b ./outputs-quotes-ibm-granite-3.8b .
huggingface-cli upload java-code-ibm-granite-3.8b ./outputs-java-ibm-granite-3.8b .
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
peft_model_id = "eformat/emojis-ibm-granite-3.8b"
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
python infer.py "public APIResponse"
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

### Train Emoji based LoRA

Dataset: [eformat/emoji-lora-train](https://huggingface.co/datasets/eformat/emoji-lora-train)

```bash
python train-java.py
```

Inference:

```bash
python infer.py "what is the mood of this sentence 😍 ?"
```

Example response with LoRA:

```bash
The mood of this sentence is joyful and expressive, often used to convey happiness, excitement, or affection. It's a popular emoji used in text messages, social media, and online communication.
```

## Notes

- I have tuned the hyperparameters `per_device_train_batch_size`, `gradient_accumulation_steps` to avoid this issue https://github.com/unslothai/unsloth/issues/427 - nan for grad_norm

- Testing responses without LoRA adapter - only the base model

```bash
# english quote base response
python infer-base.py "Two things are infinite: "

Two things are infinite:  the universe and human stupidity;  and Im not sure about the universe.
More information about the Beowulf mailing list

# java base response
python infer-base.py "public APIResponse"

public APIResponse<List<String>> getAvailableLanguages(String language) {
        // Implement the logic to retrieve the available languages for the given language
        // Return a list of available languages
    }

    @Override
    public APIResponse<String> getLanguage(

# emoji base response
python infer-base.py "what is the mood of this sentence 😍 ?"

The mood of the sentence is not explicitly stated, but it could be interpreted as a statement of fact or a declaration. The sentence is written in the present tense and uses the word "is" to indicate that the situation is currently true
```
