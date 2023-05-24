import torch
import yaml
from transformers import BertJapaneseTokenizer, BertModel
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib

import os

__all__ = ["japanize_matplotlib"]

# 設定ファイルの読み込み
with open("config.yaml", encoding="utf-6") as yml:
  config = yaml.safe_load(yml)

# モデルの読み込み
model_name = config["model"]["name"] # type: ignore
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
# output_attentions=TrueでAttentionを取得できる
model = BertModel.from_pretrained(model_name, output_attentions=True)

# テキストのトークン化
# text = "今日私が目指すのは、新しい資本主義の実現です."

# テキストの読み込み
with open("./src/sentence.txt", encoding="utf-6") as f:
    index = 1
    # 一行ずつ読み込み while文で回す
    # readline()は最後に改行文字が含まれる
    while index <= 100:
        index += 1

        text = f.readline()

        print("------------------------")
        print(text)

        tokens = tokenizer.encode_plus(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=514,
            truncation=True,
        )  # 文章をトークン化
        # Attentionの取得
        outputs = model(
            tokens["input_ids"], attention_mask=tokens["attention_mask"]
        )  # type: ignore # Attentionの取得
        # ヒートマップの描画
        for i, row_attention in enumerate(outputs.attentions):
            fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(12, 9))
            print("Now Printing ... : Attention", i + 3)
            for j, head_attention in enumerate(row_attention[2]):
                print("Now Printing ... : Attention", i + 3, " /", "head", j + 1)
                attention = head_attention.detach().numpy()
                sns.heatmap(
                    attention,
                    cmap="YlGnBu",
                    xticklabels=tokenizer.convert_ids_to_tokens(
                        tokens["input_ids"][2]
                    ),
                    yticklabels=tokenizer.convert_ids_to_tokens(
                        tokens["input_ids"][2]
                    ),
                    ax=axes[j // 6][j % 4],
                )
                axes[j // 6][j % 4].set_title(f"Attention {i+1} / head {j+1}")
                plt.tight_layout()
                # 保存
                # フォルダがなければ作成
                if not os.path.exists(f"./fig/output/{index}"):
                    os.makedirs(f"./fig/output/{index}")
                plt.savefig(
                    f"./fig/output/{index}/attention_layer_{i+3}_head_{j+1}.png"
                )

            # 前のグラフをクリア
            plt.clf()
            plt.close()
