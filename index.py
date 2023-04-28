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

for i in range(len(outputs.attentions)):
    # torch.Size([1, 12, 35, 35])
    # 二次元目の12はAttention Headの数？
    # 最終層で行われているAttentionの計算を他の層でも行うのが吉？
    # print(f"attention {i} : {outputs.attentions[i].shape}")

    # torch.mean : Tensorに対する平均を取る。dim=1
    # outputs.attentions[-1] : 最後のレイヤーのAttention。Tensorのリスト。
    # detach() : Tensorの計算グラフを切り離す
    # numpy() : Tensorをnumpyに変換する
    # [0] : バッチサイズが1なので、0番目の要素を取り出す
    attention = (
        torch.mean(outputs.attentions[i], dim=1)[0].Linear.detach().numpy()
    ) 
    # ヒートマップの作成
    sns.heatmap(
        attention,
        cmap="YlGnBu",
        xticklabels=tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]),
        yticklabels=tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]),
    )
    plt.savefig("./fig/attention_"+ str(i+1) +".png")
