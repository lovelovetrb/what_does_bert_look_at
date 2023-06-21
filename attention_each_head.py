import torch
import yaml
from transformers import BertJapaneseTokenizer, BertModel
from transformers import T5Tokenizer, RobertaForMaskedLM
from transformers import MLukeTokenizer, LukeModel
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib

import os

__all__ = ["japanize_matplotlib"]


def main(config):
    # モデルの読み込み
    model_name_list = config["model"]["name"]

    for model_name in model_name_list:
        print(f"Now Loading ... : {model_name}")
        if model_name == "cl-tohoku/bert-base-japanese":
            tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
            # output_attentions=TrueでAttentionを取得できる
            model = BertModel.from_pretrained(model_name, output_attentions=True)
            gen_attention(tokenizer,model,model_name)
        elif model_name == "rinna/japanese-roberta-base":
            # load tokenizer
            tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-roberta-base")
            tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
            # load model
            model = RobertaForMaskedLM.from_pretrained(model_name, output_attentions=True)
            gen_attention(tokenizer,model,model_name)
        elif model_name == "rinna/japanese-gpt2-medium":
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
            model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
        elif model_name == "studio-ousia/luke-japanese-base":
            tokenizer = MLukeTokenizer.from_pretrained(model_name)
            model = LukeModel.from_pretrained(model_name, output_attentions=True )
            gen_attention(tokenizer,model,model_name)

def gen_attention(tokenizer,model,model_name, nrows=3):
     # テキストの読み込み
    with open("./src/sentence.txt", encoding="utf-8") as f:
        index = 1
        # 一行ずつ読み込み while文で回す
        # readline()は最後に改行文字が含まれる
        while True:
            text = f.readline()
            if text == "":
                break

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
            gen_picture(tokens, outputs, tokenizer, model_name, index, nrows=nrows)

def gen_picture(tokens, outputs, tokenizer, model_name, index, nrows):
    # ヒートマップの描画
    for i, row_attention in enumerate(outputs.attentions):
        fig, axes = plt.subplots(nrows=nrows, ncols=4, figsize=(12, 9))
        print("Now Printing ... : Attention", i + 1)
        for j, head_attention in enumerate(row_attention[0]):
            print("Now Printing ... : Attention", i + 1, " /", "head", j + 1)
            attention = head_attention.detach().numpy()
            sns.heatmap(
                attention,
                cmap="YlGnBu",
                xticklabels=tokenizer.convert_ids_to_tokens(
                    tokens["input_ids"][0]
                ),
                yticklabels=tokenizer.convert_ids_to_tokens(
                    tokens["input_ids"][0]
                ),
                ax=axes[j // 4][j % 4],
            )
            axes[j // 4][j % 4].set_title(f"Attention {i+1} / head {j+1}")
            plt.tight_layout()
            # 保存
            # フォルダがなければ作成
            if not os.path.exists(f"./fig/output_each_model/{model_name}"):
                os.makedirs(f"./fig/output_each_model/{model_name}")
            if not os.path.exists(f"./fig/output_each_model/{model_name}/{index}"):
                os.makedirs(f"./fig/output_each_model/{model_name}/{index}")
        plt.savefig(
            f"./fig/output_each_model/{model_name}/{index}/attention_layer_{i+1}_head_{j+1}.png"
        )

        # 前のグラフをクリア
        plt.clf()
        plt.close()
    index += 1


if __name__ == "__main__":
    # 設定ファイルの読み込み
    with open("config.yaml", encoding="utf-8") as yml:
        config = yaml.safe_load(yml)
    main(config)
