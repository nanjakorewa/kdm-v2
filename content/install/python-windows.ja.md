---
title: Windows で Python 3.13 と仮想環境を用意する
pre: "1.1 "
weight: 1
---

Windows 11/10 上で Python 3.13 を導入し、`uv` を使って仮想環境の作成・パッケージ管理を行う手順です。

## 1. Python 3.13 のインストール

1. 公式サイトから 64bit 版インストーラを取得します。  
   [Python 3.13.x Windows installer](https://www.python.org/downloads/windows/) を開き、`Windows installer (64-bit)` を選択します。
2. インストーラを起動し、**Add python.exe to PATH** にチェックを入れてから `Install Now` をクリックします。
3. セットアップ完了後、PowerShell を開いてバージョンを確認します。

   ```powershell
   py -3.13 --version
   ```

   `Python 3.13.x` と表示されればインストール成功です。  
   ※ `winget install Python.Python.3.13` を利用しても同じバージョンを導入できます。

## 2. uv のインストール

`uv` は公式スクリプトでインストールできます。PowerShell を管理者権限で開き、次を実行します。

```powershell
Set-ExecutionPolicy -Scope Process Bypass
iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex
```

コマンド完了後、以下で確認します。

```powershell
uv --version
```

## 3. プロジェクト用ディレクトリと仮想環境

```powershell
mkdir C:\projects\my-app
cd C:\projects\my-app
uv venv --python 3.13 .venv
```

仮想環境をアクティブにする場合は次の通りです。

```powershell
.\.venv\Scripts\Activate.ps1
```

`uv run` を使えば有効化せずにコマンドを実行することも可能です（例：`uv run python app.py`）。

## 4. パッケージ管理

`uv` で直接ライブラリを追加できます。

```powershell
uv pip install numpy pandas
```

`requirements.txt` がある場合は同期コマンドを使います。

```powershell
uv pip sync requirements.txt
```

## 5. 仮想環境の終了

作業を終えたら次のコマンドで仮想環境を抜けることができます。

```powershell
deactivate
```

これで Windows 上に Python 3.13 + uv ベースの仮想環境が整いました。
