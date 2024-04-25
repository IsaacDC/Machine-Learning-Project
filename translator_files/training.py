import torch
from transformers import MarianMTModel, MarianTokenizer
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Load the OPUS100 dataset
language_to_train = "es"
dataset = load_dataset("Helsinki-NLP/opus-100", f"en-{language_to_train}")
dataset_size = len(dataset["train"]) // 4
reduced_dataset = dataset["train"].select(range(dataset_size))

test_dataset = dataset["test"]

# Load the pretrained OPUS-MT model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-mul"
model_path = r"model_checkpoints\fine_tuned_opus_mt_model_checkpoint.pt"  # Path to save model checkpoint
model = MarianMTModel.from_pretrained(model_name)
model = model.cuda()

tokenizer = MarianTokenizer.from_pretrained(model_name)

model.load_state_dict(torch.load(model_path))

def encode_input_str(
    text, tokenizer, seq_len
):
    # Tokenize and add special tokens
    input_ids = tokenizer.encode(
        text= text,
        return_tensors="pt",
        padding= True,
        truncation=True,
        max_length=seq_len,
    )

    return input_ids[0]

def encode_target_str(text, tokenizer, seq_len):
    token_ids = tokenizer.encode(
        text=text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )

    return token_ids[0]

def train_test():
    # Define fine-tuning parameters
    batch_size = 16  # Reduce batch size
    learning_rate = 3e-5
    num_epochs = 3
    print_freq = 100  # Print update every 100 batches
    checkpoint_freq = 1000  # Save checkpoint every 1000 batches
    n_batches = int(np.ceil(len(reduced_dataset) / batch_size))
    total_steps = num_epochs * n_batches
    n_warmup_steps = int(total_steps * 0.01)

    # Fine-tuning loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, n_warmup_steps, total_steps)


    losses = []  # List to store losses for printing updates

    def eval_model(model, gdataset, batch_size, max_iters=8):
        data_loader = torch.utils.data.DataLoader(gdataset, batch_size=batch_size, shuffle=False)
        eval_losses = []
        
        model.eval()  # Set model to evaluation mode

        for i, batch in enumerate(data_loader):
            if i >= max_iters:
                break

            input_batch = batch["input_ids"].cuda()  # Move input data to GPU
            label_batch = batch["labels"].cuda()  # Move labels to GPU

            with torch.no_grad():  # Disable gradient calculation during evaluation
                model_out = model(input_ids=input_batch, labels=label_batch)
                eval_losses.append(model_out.loss.item())

        model.train()  # Set model back to training mode

        return np.mean(eval_losses)

    for epoch_idx in range(num_epochs):
        total_loss = 0
        data_loader = torch.utils.data.DataLoader(reduced_dataset, batch_size=batch_size, shuffle=True)
        progress_bar = tqdm(enumerate(data_loader),
                            desc=f"Epoch {epoch_idx+1}/{num_epochs}", leave=False, total=len(reduced_dataset) // batch_size)

        for batch_idx, batch in progress_bar:
            input_texts = batch["translation"]["en"]
            target_texts = batch["translation"][language_to_train]
            
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
            labels = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True)
            
            inputs = {key: val.cuda() for key, val in inputs.items()}
            labels = {key: val.cuda() for key, val in labels.items()}
            
            optimizer.zero_grad()

            outputs = model(**inputs, labels=labels["input_ids"])

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            losses.append(loss.item())
            
            progress_bar.set_postfix(loss=loss.item())  # Update progress bar with current loss
            
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

        average_loss = total_loss / len(dataset["train"])
        print(f"Epoch {epoch_idx+1}/{num_epochs}, Average Loss: {average_loss:.4f}")

    # Save the fine-tuned model as a .pt file
    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    train_test()

