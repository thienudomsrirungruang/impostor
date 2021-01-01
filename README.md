# Impostor-bot

Inspired by https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313 and https://towardsdatascience.com/speak-to-the-dead-with-deep-learning-a336ef88425d.

Put your chat data into `impostor/data`

## Installation

This project was installed on conda, python version 3.9.

Pytorch installation:

`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -c=conda-forge`

Hugging Face transformer install:

`pip install transformers`

Also requires (from pip):

* matplotlib
* pyyaml
* discord.py

And from conda:
* ftfy
* spacy

## Usage
1. Make a directory with exported LINE chat logs. Remove the headers in each of the files.
2. Put this directory in the `chat_log_directory` in `config.yaml`.
3. Put your LINE username into the `user_name` field in `config.yaml`.
4. Enter a file location for the dataset in the `dataset/file` field.
5. Run `create_dataset.py`.
6. Configure training parameters in `config.yaml` (especially the `device`).
7. Run `train.py`.
8. Choose a model saved in the `checkpoints` folder and put it in `model_path` in `config.yaml`.
9. Create a discord bot.
10. Put the bot secret in `bot_token`.
11. Invite the bot to your favourite server. Link should look like
https://discord.com/oauth2/authorize?client_id=1234567890&permissions=68672&scope=bot
12. Run `bot.py`
13. Chat with your bot!
