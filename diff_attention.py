import torch
import yaml
from transformers import BertJapaneseTokenizer, BertModel
import japanize_matplotlib

import os

__all__ = ["japanize_matplotlib"]

# 設定ファイルの読み込み
with open("config.yaml", encoding="utf-8") as yml:
    config = yaml.safe_load(yml)

# モデルの読み込み
model_name = config["model"]["name"]
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
# output_attentions=TrueでAttentionを取得できる
model = BertModel.from_pretrained(model_name, output_attentions=True)

# テキストのトークン化
# text = "今日私が目指すのは、新しい資本主義の実現です."

# 空のnumpy配列を作成
attention_list = []

# テキストの読み込み
with open("./src/sentence.txt", encoding="utf-8") as f:
    index = 0
    # 一行ずつ読み込み while文で回す
    # readline()は最後に改行文字が含まれる
    while index <= 100:
        text = f.readline()
        index += 1
        print("------------------------")
        print(text)

        tokens = tokenizer.encode_plus(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=512,
            truncation=True,
        )  # 文章をトークン化
        # Attentionの取得
        outputs = model(
            tokens["input_ids"], attention_mask=tokens["attention_mask"]
        )  # Attentionの取得

        # 12 * 12 の配列
        attention = outputs.attentions[0]

        #  sum_attentionに結合
        attention_list.append(attention)

# sum_attentionのばらつきを見たい
# 12 * 12 * 100 の配列
sum_attention = torch.sum(torch.stack(attention_list), dim=0)
