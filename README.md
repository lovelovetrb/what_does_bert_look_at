## What Does Transformer look at?
A repository that contains code and results for visualizing Attention and analyzing it using the some transformer model

### models
BERT model: https://huggingface.co/cl-tohoku/bert-base-japanese
RoBERTa model: https://huggingface.co/rinna/japanese-roberta-base
GPT-2 model: https://huggingface.co/rinna/japanese-gpt2-medium
Luke model: https://huggingface.co/docs/transformers/main/en/model_doc/luke

### How to Use
1. Download reliance package
    pip
    ```
     pip install -r requirements.lock
    ```
    rye
    ```
     rye sync
    ```

2. Add model config in `attention_each_head.py`
3. Run `attention_each_head.py`

※You can describe the text to be analyzed in `src/sentence.txt`
