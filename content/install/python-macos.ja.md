---
title: macOS で Python 3.13 と仮想環境を用意する
pre: "1.3 "
weight: 3
---

macOS Sonoma / Ventura を想定して、Homebrew で Python 3.13 を導入し、`uv` を使って仮想環境を構築する手順です。

## 1. Homebrew で Python 3.13 をインストール

```bash
brew update
brew install python@3.13
```

インストール後、シムリンクを設定します。

```bash
brew link python@3.13
python3.13 --version
```

## 2. uv のインストール

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

インストール完了後、`~/.local/bin` を PATH に追加し、バージョンを確認します。

```bash
uv --version
```

## 3. プロジェクト用ディレクトリと仮想環境

```bash
mkdir -p ~/projects/my-app
cd ~/projects/my-app
uv venv --python 3.13 .venv
```

仮想環境を有効化する場合は次を実行します。

```bash
source .venv/bin/activate
```

`uv run python script.py` のように実行すれば、仮想環境を有効化せずにスクリプトを動かせます。

## 4. パッケージ管理

```bash
uv pip install numpy pandas
```

`requirements.txt` がある場合は以下で同期します。

```bash
uv pip sync requirements.txt
```

## 5. 仮想環境の終了

```bash
deactivate
```

これで macOS 上でも Python 3.13 + uv を利用した仮想環境が整いました。
