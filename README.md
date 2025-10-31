
# K_DM_book_v2

## プロジェクト概要
- 機械学習・データ解析の基本を日本語中心に解説するオンラインブックを Hugo で構築するリポジトリです。
- 章ごとの Markdown コンテンツをもとに、図表は Python スクリプトで自動生成し、静的サイトとして公開します。
- Hugo Book テーマをベースにしつつ、`layouts/` や `assets/` 内でテーマを拡張／上書きしています。

## クイックスタート
- Hugo（extended 版推奨）と Node をインストールします。
- リポジトリを取得したら `hugo server` で開発サーバを起動し、`http://localhost:1313` でプレビューできます。
- 本番ビルドは `hugo --minify` を実行して `public/` 以下に静的ファイルを生成します。
- 断続的に利用する Python スクリプトがあるため、図版を再生成する際は任意の仮想環境に必要なライブラリ（Matplotlib など）をインストールしてください。

## ディレクトリ構成
| パス | 内容 |
| --- | --- |
| `content/` | 各章の記事ソース。`basic/`（機械学習基礎）、`timeseries/`、`visualize/` 等のサブディレクトリに Markdown (`*.md`) が配置され、多言語版は拡張子前に `.ja` などの言語コードを付与。トップページや `about.ja.md` などの固定ページもここに含まれます。 |
| `assets/` | Hugo Pipes でビルドされるスタイル／スクリプト類。`book.scss` や `_*.scss` のカスタムスタイル、KaTeX 設定 (`katex.json`)、検索・Service Worker 関連 JS などが入っています。 |
| `layouts/` | テーマ上書き用テンプレート。`partials/docs/inject/head.html` や `shortcodes/` など、Hugo Book テーマをカスタマイズする HTML テンプレートが格納されています。 |
| `static/` | 生成済み／手動管理の静的アセット。`static/images/...` に各章で使用する図版が章別フォルダで整理され、`ads.txt` や Netlify 向け `_redirects` もここに置かれます。 |
| `scripts/` | 図表生成や原稿整形用のユーティリティ。詳しくは `scripts/README.py` を参照。`generate_basic_assets.py` などが Markdown のコードブロックを実行し、`static/images` に SVG を保存します。 |
| `themes/book/` | ベースとなる Hugo Book テーマ（git submodule）。必要に応じて `_vendor/` にも依存モジュールが vendoring されています。 |
| `archetypes/` | 新規コンテンツ作成時の Front Matter テンプレート。`default.md` が含まれます。 |
| `data/` | Hugo の構造化データ置き場（現在は空）。新しいデータ駆動コンポーネントを追加する際に利用します。 |
| `resources/` | Hugo ビルドキャッシュ。パイプライン生成物が格納されるため手動編集不要です。 |
| `public/` | `hugo` 実行時に生成される最終静的サイト出力。配布物なので Git 管理から除外する運用が推奨です。 |
| `.github/` | GitHub Actions など CI 設定。 |
| `.vscode/`, `.codex/` | VS Code や Codex CLI 用の開発支援設定。 |
| `config.toml`, `hugo.toml` | サイト全体の設定ファイル。環境やビルド要件に応じて Hugo が参照します。 |
| `go.mod`, `go.sum`, `_vendor/` | Hugo Modules の依存管理。テーマやプラグインをモジュールとして取り込みます。 |

## コンテンツの追加・更新フロー
- 新しい章／節を追加する場合は `content/<カテゴリ>/` 配下に Markdown を作成し、Front Matter を設定します。
- 図版が必要な場合は、Markdown の ```python ブロック内で Matplotlib 等を用いて図を生成し、`scripts/generate_*.py` を実行して `static/images/...` に出力させます。
- レイアウトやメタ情報を調整したい場合は、`layouts/partials/` や `assets/` のスタイル／JS を編集します。
- プロジェクト全体のスタイルや検索設定は `assets/` 配下のリソース、メニューや言語設定は `config.toml` / `hugo.toml` で管理しています。

## コントリビューションのヒント
- 変更後は `hugo server` でローカル確認し、図版を更新した場合は生成したファイルもコミット対象に含めます。
- Python スクリプトに依存する場合は、必要ライブラリと実行手順を PR やドキュメントに明記すると他メンバーが追従しやすくなります。
- Hugo テンプレートを編集する際は、テーマアップデートによる差分に備えて変更箇所を最小限／コメント付きで管理することを推奨します。
