---
title: Ubuntu で Python 3.13 と仮想環境を用意する
pre: "1.2 "
weight: 2
---

Ubuntu 22.04/24.04 で Python 3.13 を導入し、`uv` を使って仮想環境の作成とパッケージ管理を行う手順です。

## 1. Python 3.13 のインストール

`deadsnakes` リポジトリを追加して Python 3.13 を取得します。

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.13 python3.13-venv python3.13-dev python3.13-distutils
```

インストール後、バージョンを確認します。

```bash
python3.13 --version
```

## 2. uv のインストール

公式インストールスクリプトを利用します。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

完了後、`~/.local/bin` を PATH に追加し、バージョンを確認します。

```bash
~/.local/bin/uv --version
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

有効化せずにコマンドを実行したいときは `uv run` を使えます（例：`uv run python script.py`）。

## 4. パッケージ管理

`uv` でライブラリを導入します。

```bash
uv pip install numpy pandas
```

`requirements.txt` がある場合は次の通りです。

```bash
uv pip sync requirements.txt
```

## 5. 仮想環境の終了

作業を終えたら仮想環境を抜けます。

```bash
deactivate
```

これで Ubuntu 上でも Python 3.13 + uv を用いた環境構築が完了です。
