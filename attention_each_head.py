import torch
import yaml
from transformers import BertJapaneseTokenizer, BertModel
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib

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
text = "私が目指すのは、新しい資本主義の実現です."
tokens = tokenizer.encode_plus(
    text, return_tensors="pt", add_special_tokens=True, max_length=512, truncation=True
)  # 文章をトークン化
# Attentionの取得
outputs = model(
    tokens["input_ids"], attention_mask=tokens["attention_mask"]
)  # Attentionの取得

# ヒートマップの描画
for i, row_attention in enumerate(outputs.attentions):
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 9))
    print("Now Printing ... : Attention", i + 1)
    for j, head_attention in enumerate(row_attention[0]):
        print("Now Printing ... : Attention", i + 1, " /", "head", j + 1)
        attention = head_attention.detach().numpy()

        sns.heatmap(
            attention,
            cmap="YlGnBu",
            xticklabels=tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]),
            yticklabels=tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]),
            ax=axes[j // 4][j % 4],
        )
        axes[j // 4][j % 4].set_title(f"Attention {i+1} / head {j+1}")
    plt.tight_layout()
    plt.savefig(f"./fig/attention_layer_{i+1}_head_{j+1}.png")

    # 前のグラフをクリア
    plt.clf()
