from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
import os

model_repo = "google/mt5-small"
max_new_tokens = 20
model_path = r"model_checkpoints\model.pt"
model = AutoModelForSeq2SeqLM.from_pretrained(model_repo)

tokenizer = AutoTokenizer.from_pretrained(model_repo)

# Prepare Dataset
dataset = load_dataset("alt")

train_dataset = dataset["train"]
test_dataset = dataset["test"]

LANG_TOKEN_MAPPING = {"en": "<en>", "ja": "<jp>"}


def encode_input_str(
    text, target_lang, tokenizer, seq_len, lang_token_map=LANG_TOKEN_MAPPING
):
    target_lang_token = lang_token_map[target_lang]

    # Tokenize and add special tokens
    input_ids = tokenizer.encode(
        text=target_lang_token + text,  #'<jp>' + 'Random Sentence'
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )

    return input_ids[0]


def encode_target_str(text, tokenizer, seq_len, lang_token_map=LANG_TOKEN_MAPPING):
    token_ids = tokenizer.encode(
        text=text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )

    return token_ids[0]


def format_translation_data(translations, lang_token_map, tokenizer, seq_len=128):
    # Choose a random 2 languages for in i/o
    langs = list(lang_token_map.keys())
    input_lang, target_lang = np.random.choice(langs, size=2, replace=False)

    # Get the translations for the batch
    input_text = translations[input_lang]
    target_text = translations[target_lang]

    if input_text is None or target_text is None:
        return None

    input_token_ids = encode_input_str(
        input_text, target_lang, tokenizer, seq_len, lang_token_map
    )

    target_token_ids = encode_target_str(
        target_text, tokenizer, seq_len, lang_token_map
    )

    return input_token_ids, target_token_ids


def transform_batch(batch, lang_token_map, tokenizer):
    inputs = []
    targets = []
    for translation_set in batch["translation"]:
        formatted_data = format_translation_data(
            translation_set, lang_token_map, tokenizer, max_new_tokens
        )

        if formatted_data is None:
            continue

        input_ids, target_ids = formatted_data
        inputs.append(input_ids.unsqueeze(0))
        targets.append(target_ids.unsqueeze(0))

    batch_input_ids = torch.cat(inputs)
    batch_target_ids = torch.cat(targets)

    return batch_input_ids, batch_target_ids


def get_data_generator(dataset, lang_token_map, tokenizer, batch_size=32):
    dataset = dataset.shuffle()
    for i in range(0, len(dataset), batch_size):
        raw_batch = dataset[i : i + batch_size]
        yield transform_batch(raw_batch, lang_token_map, tokenizer)

model.load_state_dict(torch.load(model_path))

n_epochs = 5
batch_size = 16
print_freq = 50
checkpoint_freq = 1000
lr = 5e-4
n_batches = int(np.ceil(len(train_dataset) / batch_size))
total_steps = n_epochs * n_batches
n_warmup_steps = int(total_steps * 0.01)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, n_warmup_steps, total_steps)

losses = []


def eval_model(model, gdataset, max_iters=8):
    test_generator = get_data_generator(
        gdataset, LANG_TOKEN_MAPPING, tokenizer, batch_size
    )
    eval_losses = []
    for i, (input_batch, label_batch) in enumerate(test_generator):
        if i >= max_iters:
            break

        model_out = model.forward(input_ids=input_batch, labels=label_batch)
        eval_losses.append(model_out.loss.item())

    return np.mean(eval_losses)

#Training
def load_or_train_model(model_repo, model_path):
    if os.path.exists(model_path):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_repo)
        model.load_state_dict(torch.load(model_path))
        model = model.eval()
    else:
        for epoch_idx in range(n_epochs):
            # Randomize data order
            data_generator = get_data_generator(
                train_dataset, LANG_TOKEN_MAPPING, tokenizer, batch_size
            )

            for batch_idx, (input_batch, label_batch) in tqdm(
                enumerate(data_generator), total=n_batches
            ):
                optimizer.zero_grad()

                # Forward pass
                model_out = model.forward(input_ids=input_batch, labels=label_batch)

                # Calculate loss and update weights
                loss = model_out.loss
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Print training update info
                if (batch_idx + 1) % print_freq == 0:
                    avg_loss = np.mean(losses[-print_freq:])
                    print(
                        "Epoch: {} | Step: {} | Avg. loss: {:.3f} | lr: {}".format(
                            epoch_idx + 1, batch_idx + 1, avg_loss, scheduler.get_last_lr()[0]
                        )
                    )

                if (batch_idx + 1) % checkpoint_freq == 0:
                    test_loss = eval_model(model, test_dataset)
                    print("Saving model with test loss of {:.3f}".format(test_loss))
                    torch.save(model.state_dict(), model_path)

        torch.save(model.state_dict(), model_path)

    return model


test_sentence = 'Hello my name is Goku Takoshima I am japanese'
print('Raw input text:', test_sentence)

input_ids = encode_input_str(
    text = test_sentence,
    target_lang = 'ja',
    tokenizer = tokenizer,
    seq_len = model.config.max_length,
    lang_token_map = LANG_TOKEN_MAPPING)
input_ids = input_ids.unsqueeze(0)

print('Truncated input text:', tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(input_ids[0])))

output_tokens = model.generate(input_ids, num_beams=10, num_return_sequences=3)
# print(output_tokens)
for token_set in output_tokens:
  print(tokenizer.decode(token_set, skip_special_tokens=True))
