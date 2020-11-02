print("Loading model do not close the program unless unresponsive for more than 3 minutes...")

from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import inquirer



sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_LEN = 500
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
class SentimentClassifier(nn.Module):
    
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
  
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)


class_names = ['Negative', 'Neutral', 'Positive']
model = SentimentClassifier(len(class_names))
print("Loading model do not close the program unless unresponsive for more than 3 minutes...")
model.load_state_dict(torch.load('best_model_state.bin'))
model = model.to(device)
model.eval()

while True:
  questions = [
    inquirer.List('option',
                  message="Enter your review or select preset reviews or select EXIT to close the program: ",
                  choices=['Enter custom review >', 'The movie was great.', 'The movie was bad.', 'The movie was not horrible.', 'EXIT'],
              ),
  ]
  answer = inquirer.prompt(questions)

  if answer["option"]=="EXIT":
    break
  review_text = answer["option"]
  if answer["option"]=="Enter custom review >":
    review_text = input("Enter your review: ")

  encoded_review = tokenizer.encode_plus(
    review_text,
    max_length=MAX_LEN,
    add_special_tokens=True,
    return_token_type_ids=False,
    padding=True,
    return_attention_mask=True,
    return_tensors='pt',
  )
  input_ids = encoded_review['input_ids'].to(device)
  attention_mask = encoded_review['attention_mask'].to(device)

  output = model(input_ids, attention_mask)
  _, prediction = torch.max(output, dim=1)

  probs = output.detach().cpu().data.numpy()[0]
  min_max = lambda v:(v - probs.min()) / (probs.max() - probs.min())
  probs = np.array([min_max(xi) for xi in probs])
  sum_probs=sum(probs)
  probs = np.array([x/sum_probs for x in probs])
  print("===============================================")
  print("Probabilities:")
  print(class_names[0]+":{}".format(probs[0]))
  print(class_names[1]+":{}".format(probs[1]))
  print(class_names[2]+":{}".format(probs[2]))
  print()
  print(f'Review text: {review_text}')
  print(f'Sentiment  : {class_names[prediction]}')
  print("===============================================")
  print()
  pred_df = pd.DataFrame({
    'class_names': class_names,
    'values': probs
  })
  sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
  plt.ylabel('sentiment')
  plt.xlabel('probability')
  plt.xlim([-1, 1]);
  plt.title("Sentiment label probabilities");
  plt.show()