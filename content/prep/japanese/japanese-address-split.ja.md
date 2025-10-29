---
title: "住所を都道府県・市区町村に分割する"
pre: "3.7.6 "
weight: 6
title_suffix: "一覧から最長一致で行政区を抽出"
---

請求書やアンケートの住所から都道府県・市区町村を取り出すと、地域分析や配送ロジックが組みやすくなります。行政区の一覧を持っておき、最長一致で切り出すのがシンプルで堅牢です。

```python
PREFECTURES = [
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
    "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
    "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県",
    "岐阜県", "静岡県", "愛知県", "三重県",
    "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県",
    "鳥取県", "島根県", "岡山県", "広島県", "山口県",
    "徳島県", "香川県", "愛媛県", "高知県",
    "福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県",
    "沖縄県",
]

def split_address(address: str) -> dict[str, str]:
    prefecture = ""
    for pref in PREFECTURES:
        if address.startswith(pref):
            prefecture = pref
            break

    rest = address[len(prefecture):]
    city = ""
    for suffix in ("市", "区", "町", "村"):
        if suffix in rest:
            idx = rest.find(suffix)
            city = rest[: idx + 1]
            rest = rest[idx + 1 :]
            break

    return {
        "prefecture": prefecture,
        "city": city,
        "remaining": rest.strip(),
    }


samples = [
    "東京都千代田区霞が関1-1-1 ○○ビル3F",
    "大阪府大阪市北区梅田3丁目1-1",
    "福岡県宗像市田熊4-1-1",
]

for s in samples:
    print(split_address(s))
```

### 拡張のヒント
- 政令指定都市や郡部 (`○○郡△△町`) に対応するには、総務省の最新データ（JIS コード）をダウンロードし、郡・町レベルまでのリストを作ると確実です。
- 町名以降をジオコーディングしたい場合は、`geopy` や自治体の地理院 API と組み合わせて緯度経度を付与します。
- 郵便番号で正規化する場合は Japan Post の CSV をマスタにしておくと、誤字の修正やローマ字出力も同時に行えます。
