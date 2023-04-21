import torch
import yaml
from transformers import BertTokenizer, BertModel
import seaborn as sns
import matplotlib.pyplot as plt

with open('config.yaml', 'r') as yml:
    config = yaml.safe_load(yml)

model_name = BertModel.from_pretrained(config['model']['name'])
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, output_attentions=True)

# テキストのトークン化
text = "私は昨日友達が野球の試合を見に行った話を聞きました。"
tokens = tokenizer.encode_plus(
    text, return_tensors='pt', add_special_tokens=True, max_length=512, truncation=True)

# Attentionの取得
outputs = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])
attention = torch.mean(outputs.attentions[-1], dim=1)[0].detach().numpy()

# ヒートマップの作成
sns.heatmap(attention, cmap="YlGnBu", xticklabels=tokenizer.convert_ids_to_tokens(
    tokens['input_ids'][0]), yticklabels=tokenizer.convert_ids_to_tokens(tokens['input_ids'][0]))
plt.show()
