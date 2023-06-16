import yaml

# 設定ファイルの読み込み
with open("config.yaml", encoding="utf-8") as yml:
    config = yaml.safe_load(yml)

# モデルの読み込み
model_name = config["model"]["name"]
print(model_name[0])