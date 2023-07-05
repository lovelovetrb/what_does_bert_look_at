import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BertJapaneseTokenizer, BertModel

# tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-7b")
# model = AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-7b", output_attentions=True)

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium", use_fast=False)
tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium", output_attentions=True)

# tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
# model = BertModel.from_pretrained("cl-tohoku/bert-base-japanese", output_attentions=True)

text = "今日は暖かい日で、心地よい風が吹いています。"
tokens = tokenizer.encode_plus(
    text,
    return_tensors="pt",
    add_special_tokens=True,
    max_length=514,
    truncation=True,
)
outputs = model(
    tokens["input_ids"], attention_mask=tokens["attention_mask"]
)

print(tokens)
print(outputs.attentions)
print(len(outputs.attentions[0][0]))

# decode ids to tokens
for token in tokens["input_ids"][0]:
    decoded = tokenizer.decode(token)
    print(decoded)
