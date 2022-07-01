# slack-export-parser

Convert a Slack export into a prompt-and-completion pandas DataFrame. Purpose is to create a dataset to [fine tune a model like GTP-3](https://wandb.ai/borisd13/GPT-3/reports/Fine-Tuning-Tips-and-Exploration-on-OpenAI-s-GPT-3---VmlldzoxNDYwODA2).

```sh
# create a virtual environment

# install the package
python -m pip install /path/to/slack-export-parser

# get help
python slack_export_parser/parse_prompt_completion.py --help

python slack_export_parser/parse_prompt_completion.py --export_dir="My Export Folder" --display_name=matt --n_prior=2 --prepend_channel=True --prepend_sender=True --sender_type=display_name
```

Args:

- `export_dir`: path to the uncompressed export directory
- `display-name`: username to use for the completion / response
- `n_prior`: number of messages prior to the completion to include
- `prepend_channel`: whether to prepend the channel name to the prompt
- `prepend_sender`: whether to prepend the sender to the prompt
- `sender_type`: `display_name` or `user`
