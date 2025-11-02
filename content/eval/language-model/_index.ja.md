---

title: 言語モデルの評価指標

weight: 6

chapter: true

not_use_colab: true

not_use_twitter: true

pre: "<b>4.6 </b>"

---



# 言語モデルの評価指標





言語モデルや要約モデルを評価する際に使われる主要指標を、なかのてっぺいさんの記事「[LLM & NLP 評価指標のまとめ](https://zenn.dev/nakano_teppei/articles/be39e5a03eef69)」をベースに整理しました。タスクに合った指標を選べるように、**n-gram ベース**・**埋め込みベース**・**LLM 代替評価** の 3 つに分類しています。



## 1. n-gram ベース指標（語彙一致を重視）



| 指標 | 何を見るか | 特徴 |

| ---- | ---------- | ---- |

| **BLEU** | 1〜4-gram の precision と短縮ペナルティ | 機械翻訳の定番。短い出力や語順の違いに弱い。 |

| **ROUGE-1/2/L** | n-gram の recall / LCS の長さ | 要約でよく使われる。参照が複数あると実用的。 |

| **METEOR** | Unigram マッチ（同義語・語形変化を考慮） | Recall 志向。WordNet を利用する英語向け実装が多い。 |

| **chrF++** | 文字 n-gram の F-score | サブワード言語でも頑健。chrF に単語 n-gram を加えた改良版。 |



### 使い分けのヒント

- **BLEU**: 既存ベンチマークとの比較を重視したいとき（翻訳など）。ラベルが複数あるなら smoothing を設定しよう。

- **ROUGE**: 抽出型/要約タスクで recall を重視したいとき。`rouge-score` パッケージで手軽に計算できる。

- **METEOR**: 意味の近さも加味したい英語タスクで有効。ただし日本語のリソースは少なめ。

- **chrF++**: 形態素分割が難しい言語（日本語・中国語など）での翻訳評価に向く。



```python

# 例: sacrebleu で BLEU / chrF++ を計算

import sacrebleu



references = [["今日は良い天気です。"]]

candidate = "今日はとてもいい天気だ。"



bleu_score = sacrebleu.corpus_bleu([candidate], references)

chrf_score = sacrebleu.corpus_chrf([candidate], references)

print(bleu_score.score, chrf_score.score)

```python

# 2. 埋め込みベース指標（意味的一致を重視）



| 指標 | モデル | メモ |

| ---- | ------ | ---- |

| **BERTScore** | BERT / RoBERTa などのトークン埋め込み | F1 スコアで重み付き類似度を算出。意味的類似に強い。 |

| **MoverScore** | Word Mover’s Distance の改良版 | 低頻度語も重視。計算コストは高め。 |

| **BLEURT** | RoBERTa ベースを人手評価で再学習 | 小規模データでも高い相関。事前学習モデルをダウンロードして利用。 |

| **COMET** | 多言語 Transformer + 事前学習済み QA 損失 | 翻訳ベンチマークで高い相関。専用 CLI・API が用意されている。 |

| **QAEval / ParaScore** | 質問生成・回答で意味一致を測る | QA の整合性を通して品質を評価。準備コストが高い。 |



### 使い分けのヒント

- BERTScore は Hugging Face の `bert-score` パッケージで簡単に呼び出せ、英語以外のモデル（mBERT）にも対応。

- BLEURT や COMET は **学習済みチェックポイントを取得** して利用する。人手評価との相関が高い分、前処理と推論コストは増える。

- 意味的な正しさを優先したいが計算資源が限られている場合は **BERTScore → BLEURT/COMET** の順に検討するとよい。



```python

from bert_score import score



cands = ["今日はとてもいい天気だ。"]

refs = ["今日は良い天気です。"]

P, R, F1 = score(cands, refs, lang="ja", model_type="cl-tohoku/bert-base-japanese")

print(F1.mean().item())

```python

# 3. LLM を用いた評価（LLM-as-a-Judge）



| 手法 | 概要 | 注意点 |

| ---- | ---- | ---- |

| **LLM 直接採点** | GPT-4 などに「参照と候補を渡し採点させる」 | プロンプト設計が重要。バイアスと一貫性に注意。 |

| **LLM + ルーブリック** | 評価基準（流暢さ・忠実度など）を明示し、各項目を採点させる | 評価粒度が上がるがコストも増える。 |

| **LLM + 自動 QA** | LLM に質問生成と回答整合性チェックをさせる | QAEval の LLM 版。長文要約との相性が良い。 |



### ガイドライン

- 透明性を確保するため、**評価プロンプトとシステムメッセージを公開** しレビュー可能にする。

- 再現性を担保するため、**同じ温度・乱数設定で複数回実行**し平均を取る。

- 自動指標だけでなく、**少量でも人手評価と併用**すると品質評価への信頼度が上がる。



## チェックリスト（指標選定のフローチャート）



1. **評価するタスクは？**

   - 翻訳・要約 → n-gram 指標も計算し、ベースラインを確保。

   - 自由生成 / 対話 → 埋め込み系や LLM 評価で意味的整合性を確認。

2. **参照データは十分ある？**

   - ある → n-gram 指標 + 埋め込み指標。

   - 少ない → LLM を補助的に利用。

3. **コストと再現性のバランス**

   - 軽量で高速 → sacreBLEU / BERTScore。

   - 高精度重視 → BLEURT / COMET / GPT-judge などを併用。



## 参考リンク

- BLEU: [SacreBLEU](https://github.com/mjpost/sacrebleu)

- ROUGE: [`rouge-score` パッケージ](https://pypi.org/project/rouge-score/)

- BERTScore: [GitHub - bert-score](https://github.com/Tiiiger/bert_score)

- BLEURT: [Google Research BLEURT](https://github.com/google-research/bleurt)

- COMET: [Unbabel COMET](https://github.com/Unbabel/COMET)

- LLM-as-a-Judge: [OpenAI evals 公式サンプル](https://github.com/openai/evals) など



指標を組み合わせて評価設計を行い、人手評価との相関を確認しながら、目的に合った評価パイプラインを構築してください。

