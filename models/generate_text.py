# Load packages
import torch
import pandas as pd
import numpy as np
from torch import nn
from transformers import GPT2Tokenizer, GPT2Config, GPT2Model, GPT2PreTrainedModel
from torch.optim import AdamW
from tqdm import tqdm
from torch.nn import functional as F
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_built() else 'cpu'

# Load data
reviews = pd.read_pickle("../data/user_review.pkl")
text_corpus = [f"{txt} <|endoftext|>" for i, txt in enumerate(reviews["Comment"]) if txt != '']

######################################################################
####### Full credit for the model code goes to Ruben Winastwan #######
# https://towardsdatascience.com/text-generation-with-gpt-092db8205cad
######################################################################

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<|pad|>')

class GPT2_Model(GPT2PreTrainedModel):

    def __init__(self, config):

        super().__init__(config)

        self.transformer = GPT2Model.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<|pad|>')

        self.transformer.resize_token_embeddings(len(tokenizer))

        self.lm_head = nn.Linear(config.n_embd, len(tokenizer), bias=False)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        x = self.transformer(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        x = self.lm_head(x)

        return x

######################################################################

class ProcessedData(torch.utils.data.Dataset):

  def __init__(self, input_data, tokenizer, gpt2_type="gpt2", max_length=300):

    self.texts = [tokenizer(data, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
                    for data in input_data]

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    return self.texts[idx]

######################################################################

class CrossEntropyLossFunction(nn.Module):

    def __init__(self):

        super(CrossEntropyLossFunction, self).__init__()
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, lm_logits, labels):

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

######################################################################

def train(model, tokenizer, train_data, epochs, learning_rate, epsilon=1e-8):

    train = ProcessedData(train_data, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)

    optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )
    criterion = CrossEntropyLossFunction()#.to(device)
    #model = model.to(device)

    best_loss = 1000

    for epoch_i in range(0, epochs):

        total_train_loss = 0
        total_val_loss = 0
        avg_train_loss_list = []
        for train_input in tqdm(train_dataloader):

            mask = train_input['attention_mask']#.to(device)
            input_id = train_input['input_ids']#.to(device)

            outputs = model(input_id,
                            attention_mask = mask,
                            token_type_ids=None
                            )

            loss = criterion(outputs, input_id)

            batch_loss = loss.item()
            total_train_loss += batch_loss

            loss.backward()
            optimizer.step()
            model.zero_grad()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_loss_list.append(avg_train_loss)

        print(f"Epoch: {epoch_i}, Avg train loss: {np.round(avg_train_loss,2)}")
    return avg_train_loss_list

epochs = 35
learning_rate = 1e-5
configuration = GPT2Config()
gptmodel = GPT2_Model(configuration)#.to(device)

avg_train_loss_list = train(gptmodel, tokenizer, text_corpus, epochs, learning_rate)

plt.plot(avg_train_loss_list)
plt.title("Average Training Loss")
plt.savefig("../figures/text_generation_loss.png", bbox_inches='tight')

######################################################################

def generate(idx, max_new_tokens, context_size, tokenizer, model, top_k=10, top_p=0.95):

        for _ in range(max_new_tokens):
            if idx[:,-1].item() != tokenizer.encode(tokenizer.eos_token)[0]:
                idx_cond = idx[:, -context_size:]
                logits = model(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                sorted_probs, indices = torch.sort(probs, descending=True)
                probs_cumsum = torch.cumsum(sorted_probs, dim=1)
                sorted_probs, indices = sorted_probs[:, :probs_cumsum[[probs_cumsum < top_p]].size()[0] + 1], indices[:, :probs_cumsum[[probs_cumsum < top_p]].size()[0] +1]
                sorted_probs, indices = sorted_probs[:,:top_k], indices[:,:top_k]
                sorted_probs = F.softmax(sorted_probs, dim=-1)
                idx_next = indices[:, torch.multinomial(sorted_probs, num_samples=1)].squeeze(0)
                idx = torch.cat((idx, idx_next), dim=1)
            else:
                break

        return idx

######################################################################

gptmodel.eval()

prompt = "Burgers and donuts are"
generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
#generated = generated.to(device)

sample_outputs = generate(generated,
                         max_new_tokens=100,
                         context_size=300,
                         tokenizer=tokenizer,
                         model=gptmodel,
                         top_k=10,
                         top_p=0.95)

for i, sample_output in enumerate(sample_outputs):
    print(f"{tokenizer.decode(sample_output, skip_special_tokens=True)}")