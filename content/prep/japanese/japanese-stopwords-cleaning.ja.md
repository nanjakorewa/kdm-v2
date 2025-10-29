---
title: "日本語ストップワードを除去する"
pre: "3.7.8 "
weight: 8
title_suffix: "頻出する助詞・補助動詞を手早くフィルタ"
---

日本語の文章をキーワード抽出や TF-IDF で扱う際、助詞や補助動詞を除外すると特徴語が見えやすくなります。形態素解析を使わない簡易な方法として、正規化＋ストップワード辞書の除去を用意しておくと便利です。

```python
import re
import unicodedata

STOPWORDS = {
    "する", "なる", "いる", "ある", "こと", "もの",
    "これ", "それ", "あれ", "ここ", "そこ", "あそこ",
    "ため", "よう", "さん", "して", "した", "です", "ます",
    "から", "まで", "ので", "なら", "そして", "しかし",
}

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.lower())
    text = re.sub(r"[0-9０-９]+", "0", text)
    return text

def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[ぁ-んァ-ン一-龥]+", text)

text = "このサービスは使いやすくて、ここまで丁寧に対応してくれるのは初めてでした。"
normalized = normalize_text(text)
tokens = simple_tokenize(normalized)
clean_tokens = remove_stopwords(tokens)
print(clean_tokens)
```

### もうひと工夫するなら
- ストップワード辞書は業種ごとの頻出語（例: 「お客様」「御社」など）を追加してカスタマイズします。
- 形態素解析が利用できる環境なら `SudachiPy` や `Janome` を組み合わせ、品詞情報で助詞・助動詞を除外するとさらに精度が上がります。
- 記号や英数字が多いレビューでは、`re.findall` のパターンを調整して英単語や記号も同時にフィルタすると効果的です。
