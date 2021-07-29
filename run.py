# Like this is literally for fun.
# it's also 11:30 so I should go to bed.

# but no!
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, get_cosine_schedule_with_warmup
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from torch.utils.data import DataLoader

import random
import wandb
import torch
import tqdm
import csv
import uuid

config_defaults = dict(
    epochs = 5,
    lr = 5e-5,
    warmup = 100,
    max_length = 64,
    base_model = "facebook/bart-base",
    batch_size = 28
)

run = wandb.init(project='replier2', entity='jemoka', config=config_defaults)
config = wandb.config

EPOCHS = config.epochs
LR = config.lr
WARMUP = config.warmup
MAX_LENGTH = config.max_length
BASE_MODEL = config.base_model
BATCH_SIZE = config.batch_size

tokenizer = BartTokenizer.from_pretrained(BASE_MODEL)

I = []
O = []

with open("./movie_replies_long.csv", "r") as df:
    reader = csv.reader(df, delimiter='±')
    for row in tqdm.tqdm(reader):
        I.append(row[-2])
        O.append(row[-1])

class ReplierDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, data, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data

    def __getitem__(self, idx):
        tokenizer = self.tokenizer
        max_length = self.max_length

        input_string = self.data[0][idx]
        output_string = self.data[1][idx]

        try: 
            if output_string[-1] not in ['.', '?', '>', '!', '"']:
                return self.__getitem__(random.randint(0, len(self)-1))
        except IndexError:
            return self.__getitem__(random.randint(0, len(self)-1))
 
        input_tokenized = tokenizer.encode(input_string)
        output_tokenized = tokenizer.encode(output_string)

        if len(output_tokenized) > max_length or len(input_tokenized) > max_length:
            return self.__getitem__(random.randint(0, idx))

        input_encoded = input_tokenized + [tokenizer.pad_token_id for _ in range(max_length-len(input_tokenized))]

        output_encoded = output_tokenized + [-100 for _ in range(max_length-len(output_tokenized))]

        input_mask = [1 for _ in range(len(input_tokenized))] + [0 for _ in range(max_length-len(input_tokenized))]

        if len(input_encoded) > max_length:
            return self.__getitem__(random.randint(0, len(self)-1))

        return {"input_data": torch.LongTensor(input_encoded), "output_data": torch.LongTensor(output_encoded), "input_mask": torch.LongTensor(input_mask)}

    def __len__(self):
        return len(self.data[0])-1

dataset = ReplierDataset(tokenizer, [I,O], MAX_LENGTH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = BartForConditionalGeneration.from_pretrained(BASE_MODEL)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)

model.train()
run.watch(model)

optim = AdamW(model.parameters(), lr=LR)
scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps = WARMUP, num_training_steps = len(I)*EPOCHS)

max_acc = 0
avg_acc = 0

max_bleu = 0
avg_bleu = 0

epoch = 0
steps = 0

modelID = str(uuid.uuid4())[-5:]
smoothie = SmoothingFunction().method4

while epoch < EPOCHS:
    databatched_loader = tqdm.tqdm(loader)

    for i, chicken in enumerate(databatched_loader):
        if (i % 1000 == 0 and i != 0):
            tokenizer.save_pretrained(f"./training/bart-replier-{modelID}:ROUTINE::{epoch}:{i}")
            model.save_pretrained(f"./training/bart-replier-{modelID}:ROUTINE::{epoch}:{i}")

        input_data = chicken['input_data'].to(device)
        output_data = chicken['output_data'].to(device)
        attention_mask = chicken['input_mask'].to(device)

        result = model(input_data, attention_mask=attention_mask, labels=output_data)
        logits = result["logits"]
        loss = result["loss"]

        databatched_loader.set_description(f'{modelID} loss: {loss}')
        databatched_loader.refresh()
    
        loss.backward()

        optim.step()
        optim.zero_grad()

        scheduler.step()

        oneAnswer = torch.argmax(logits[0], dim=1)
        answer_tokens = tokenizer.convert_ids_to_tokens(oneAnswer)

        targetSec = output_data[0]

        t = targetSec[targetSec!=-100].size(0)
        c = (oneAnswer[:t] == targetSec[targetSec!=-100]).sum().item()
        w = (oneAnswer[:t] != targetSec[targetSec!=-100]).sum().item()

        acc = c/t
        avg_acc = (avg_acc+acc)/2
        max_acc = max(max_acc, acc)

        try: 
            answer_tokens_clear = [a for a in answer_tokens[0:answer_tokens.index("</s>")+1] if a != tokenizer.pad_token]
        except ValueError:
            answer_tokens_clear = [a for a in answer_tokens[0:] if a != tokenizer.pad_token]

        answer = tokenizer.convert_tokens_to_string(answer_tokens_clear)

        desiredAnswer_tokens = list(filter(lambda x:x, tokenizer.convert_ids_to_tokens(targetSec)))
        desiredAnswer = tokenizer.convert_tokens_to_string(desiredAnswer_tokens)

        inputWord_tokens = [a for a in tokenizer.convert_ids_to_tokens(input_data[0]) if a != tokenizer.pad_token]
        inputWord = tokenizer.convert_tokens_to_string(inputWord_tokens)

        try: 
            bleu = sentence_bleu([desiredAnswer_tokens], answer_tokens_clear, smoothing_function=smoothie)
        except ValueError:
            continue

        avg_bleu = (avg_bleu+bleu)/2
        max_bleu = max(max_bleu, bleu)

        if (i % 40 == 0):
            try: 
                run.log({"loss": loss.item(),
                         "accuracy": acc,
                         "bleu": bleu,
                         "input": wandb.Html(inputWord[3:-4]),
                         "logits": wandb.Histogram(logits[0].detach().cpu()),
                         "output": wandb.Html(answer[3:-4]),
                         "target": wandb.Html(desiredAnswer[3:-4])
                       })

                run.summary["max_accuracy"] = max_acc
                run.summary["avg_accuracy"] = avg_acc

                run.summary["max_bleu"] = max_bleu
                run.summary["avg_bleu"] = avg_bleu
                
            except IsADirectoryError:
                print("um.")

        steps += 1

    epoch += 1

