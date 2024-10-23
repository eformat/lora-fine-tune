# lora-fine-tune

Fine tune a GPT Lora
- https://dataman-ai.medium.com/fine-tune-a-gpt-lora-e9b72ad4ad3

`python train.py`

```bash
{'loss': 2.1376, 'grad_norm': 0.6898021101951599, 'learning_rate': 4.000000000000001e-06, 'epoch': 1.26}       
{'loss': 1.6841, 'grad_norm': 0.8817117214202881, 'learning_rate': 2.0000000000000003e-06, 'epoch': 1.27}      
{'loss': 1.8884, 'grad_norm': 0.5318072438240051, 'learning_rate': 0.0, 'epoch': 1.28}                         
{'train_runtime': 546.4885, 'train_samples_per_second': 5.856, 'train_steps_per_second': 0.366, 'train_loss': 1.9254487091302872, 'epoch': 1.28}
100%|████████████████████████████████████████████████████████████████████████| 200/200 [09:06<00:00,  2.73s/it]
```

```bash
tree outputs/
outputs/
├── adapter_config.json
├── adapter_model.safetensors
├── checkpoint-200
│   ├── adapter_config.json
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

`python infer.py`

```bash
Loading checkpoint shards: 100%|█████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.21s/it]

Two things are infinite:  the universe and human stupidity; and I'm not sure about the universe.  -Albert Einstein
I'm not sure about the universe either.
```
