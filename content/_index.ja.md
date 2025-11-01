---
title: "トップページ"
---

## K_DM Bookへようこそ
K_DM Bookは、データ分析・機械学習の学習ルートを実務レベルまで伴走する「読みながら手を動かせる」ドキュメントサイトです。数学の基礎からモデル運用、可視化やアプリ実装まで、体系的にまとめたチュートリアルとコードスニペットを公開しています。

- **段階的に学べる構成** — 「基礎 → 応用 → 実践」へ自然にステップアップできるよう章立てを整理。
- **動くコードが手元に残る** — Notebook や Python スクリプトを付属し、記事そのままに再現できるようにしています。
- **現場視点のTips** — 評価指標・前処理・可視化のベストプラクティスを、図解とチェックリスト付きで紹介。

## 学び方のおすすめルート
1. **Basics（基礎）** — 回帰・分類・クラスタリングなど主要アルゴリズムを概念と実装の両面から理解。
2. **Prep / Evaluation** — 特徴量設計や評価指標を押さえ、モデル改善の視点を身につける。
3. **Timeseries / Finance / WebApp** — ドメイン別の応用事例で、実務への展開方法を学ぶ。
4. **Visualize** — 結果の伝え方を磨くグラフカタログ。実務レポートやダッシュボード構築のヒントも併載。

## コンテンツマップ
{{< mermaid >}}
flowchart TD
  subgraph fundamentals["基礎フェーズ<br>(Fundamentals)"]
    A1[数学基礎<br>線形代数・確率・統計]
    A2[環境構築<br>Python / ライブラリ]
    A3[データ理解<br>EDA・可視化の基礎]
  end

  subgraph modeling["モデル構築フェーズ<br>(Modeling Basics)"]
    B1[回帰<br>Regression]
    B2[分類<br>Classification]
    B3[クラスタリング<br>Clustering]
    B4[次元削減<br>Dimensionality Reduction]
    B5[アンサンブル<br>Ensemble]
  end

  subgraph evaluation["評価・改善フェーズ<br>(Evaluation & Tuning)"]
    C1[評価指標<br>Metrics]
    C2[ハイパラ調整<br>Hyperparameter Tuning]
    C3[特徴量エンジニアリング<br>Feature Engineering]
    C4[モデル解釈<br>Model Explainability]
  end

  subgraph communication["可視化・共有フェーズ<br>(Communication)"]
    D1[可視化カタログ<br>Visualize Section]
    D2[レポート整備<br>Storytelling]
    D3[チーム連携Tips<br>Artifacts / Checklist]
  end

  subgraph deployment["応用・展開フェーズ<br>(Applied / Deployment)"]
    E1[時系列分析<br>Timeseries]
    E2[金融データ活用<br>Finance]
    E3[Webアプリ実装<br>Flask / Gradio]
    E4[運用自動化<br>Monitoring & Ops]
  end

  fundamentals --> modeling
  modeling --> evaluation
  evaluation --> communication
  evaluation --> deployment
  communication --> deployment

  click A2 "/install/" "環境構築ガイドへ"
  click A3 "/prep/" "データ理解の章へ"
  click B1 "/basic/regression/" "回帰セクションを開く"
  click B2 "/basic/classification/" "分類セクションを開く"
  click B3 "/basic/clustering/" "クラスタリングセクションを開く"
  click B4 "/basic/dimensionality_reduction/" "次元削減セクションを開く"
  click B5 "/basic/ensemble/" "アンサンブルセクションを開く"
  click C1 "/eval/" "評価指標の章へ"
  click C3 "/prep/feature_selection/" "特徴量エンジニアリングへ"
  click D1 "/visualize/" "可視化カタログへ"
  click E1 "/timeseries/" "時系列分析の章へ"
  click E2 "/finance/" "金融データ活用の章へ"
  click E3 "/webapp/" "Webアプリ実装の章へ"
{{< /mermaid >}}

図の順に読み進めると、モデル開発の全体像を俯瞰しながら必要な知識を埋めていけます。各ノードはサイト内のセクションに対応し、関連記事やNotebookへのリンクをまとめています。

## 付属リソースとサポート
- **Notebook / コード** — `scripts/` のユーティリティと `data/` のサンプルデータで記事をすぐに再現可能。
- **更新情報** — 新着記事はトップタイムラインと RSS（`/index.xml`）で配信。
- **フィードバック** — 誤記や改善点があれば [Issue受付フォーム](https://kdm.hatenablog.jp/entry/issue) や X / Twitter（下記）までお知らせください。

## 運営SNS
- <a href="https://www.youtube.com/@K_DM" style="color:#FF0000;"><i class="fab fa-fw fa-youtube"></i> YouTube — 解説動画・ライブ講義</a>
- <a href="https://twitter.com/_K_DM" style="color:#1DA1F2;"><i class="fab fa-fw fa-twitter"></i> X（旧Twitter） — 更新情報・短いTips</a>

プライバシーポリシーは [こちら](https://kdm.hatenablog.jp/privacy-policy) をご確認ください。コンテンツを活用し、ぜひ自身のプロジェクトへ発展させてください。
