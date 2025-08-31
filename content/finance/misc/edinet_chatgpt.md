---
title: "ChatGPT"
pre: " 7.4.5 "
weight: 33
searchtitle: "ChatGPTに決算の内容を質問するシステムをPythonで作成する"
---


{{% youtube "2mZeYEy_r_8" %}}

## ChagGPTで決算書についての質問をしてみる
### ChatGPTを使用する

<div class="pagetop-box">
    <p>四半期報告書のデータをPythonで取得し、ChatGPTで質問できるようにします！データ取得からChatGPTへの質問まですべてPython上で実行することができます。</p>
</div>


{{% notice document %}}
{{% /notice %}}


```python
import os
from openai import OpenAI


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "あなたは日本企業の決算を分析するアナリストです。"},
        {"role": "user", "content": "日本郵船とはどのような会社ですか？"},
    ],
)
print(response.choices[0].message.content)
```

    日本郵船株式会社（Nippon Yusen Kabushiki Kaisha、NYK Line）は、日本を代表する大手海運会社です。同社は、海上コンテナ輸送、液化天然ガス（LNG）船、タンカー、バルク船、自動車輸送船などの幅広い海運サービスを提供しています。また、同社は国際的な物流サービスも展開しており、陸上輸送や倉庫管理、カスタムズクリアランスなども提供しています。日本郵船は、日本の国際貿易や産業に不可欠な存在であり、世界的にも高い評価を受けている海運会社の一つです。
    

## EDINETを使用する



```python
import os
import time
import zipfile

import pandas as pd
import requests

API_ENDPOINT = "https://disclosure.edinet-fsa.go.jp/api/v2"  # v2を使用する


def save_csv(docID, type=5):
    """EDINETからデータを取得してフォルダに保存する

    Args:
        docID (str): DocID
    """
    assert type in [1, 2, 3, 4, 5], "typeの指定が間違っている"
    if type == 1:
        print(f"{docID}のXBRLデータを取得中")
    elif type == 2:
        print(f"{docID}のpdfデータを取得中")
    elif type in {3, 4}:
        print(f"{docID}のデータを取得中")
    elif type == 5:
        print(f"{docID}のcsvデータを取得中")
        time.sleep(5)

    r = requests.get(
        f"{API_ENDPOINT}/documents/{docID}",
        {
            "type": type,
            "Subscription-Key": os.environ.get("EDINET_API_KEY"),
        },
    )

    if r is None:
        print("データの取得に失敗しました。csvFlag==1かどうか確認してください。")
    else:
        os.makedirs(f"{docID}", exist_ok=True)
        temp_zip = "uuid_89FD71B5_CD7B_4833-B30D‗5AA5006097E2.zip"

        with open(temp_zip, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)

        with zipfile.ZipFile(temp_zip) as z:
            z.extractall(f"{docID}")

        os.remove(temp_zip)
```

関西ペイントの『四半期報告書－第160期第3四半期(2023/10/01－2023/12/31) 』の書類を取得します。


```python
save_csv("S100SRKD", type=1)
```

    S100SRKDのXBRLデータを取得中
    


```python
from edinet_xbrl.edinet_xbrl_parser import EdinetXbrlParser
from IPython.display import HTML

parser = EdinetXbrlParser()
edinet_xbrl_object = parser.parse_file(
    r"S100SRKD\XBRL\PublicDoc\jpcrp040300-q3r-001_E00893-000_2023-12-31_01_2024-02-09.xbrl"
)

key = "jpcrp_cor:ManagementAnalysisOfFinancialPositionOperatingResultsAndCashFlowsTextBlock"
context_ref = "FilingDateInstant"
management_analysis_description = edinet_xbrl_object.get_data_by_context_ref(
    key, context_ref
).get_value()

HTML(management_analysis_description.text)
```

    C:\Users\nanja-win-ms\miniconda3\envs\py39\lib\site-packages\bs4\builder\__init__.py:545: XMLParsedAsHTMLWarning: It looks like you're parsing an XML document using an HTML parser. If this really is an HTML document (maybe it's XHTML?), you can ignore or filter this warning. If it's XML, you should know that using an XML parser will be more reliable. To parse this document as XML, make sure you have the lxml package installed, and pass the keyword argument `features="xml"` into the BeautifulSoup constructor.
      warnings.warn(
    





<h3>２【経営者による財政状態、経営成績及びキャッシュ・フローの状況の分析】</h3>
<p style="margin-left: 24px; text-align: justify; text-justify: inter-ideograph">　文中の将来に関する事項は、当四半期連結会計期間の末日現在において判断したものであります。</p>
<p style="margin-left: 24px; text-align: justify; text-justify: inter-ideograph">(1）経営成績の状況</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">　当期における世界経済は、供給制約が概ね解消されインフレ率も鈍化の傾向が見られるものの、地政学リスクは依然高まったままで基調的な物価上昇圧力は根強く、欧米を中心に金融引き締めが継続しており、その回復ペースは鈍化しております。そのような状況下、中国においては不動産市況の停滞の影響もありゼロコロナ政策解除後の景気回復は緩慢なペースにとどまっています。欧州においては物価高や利上げによる金融引き締めが景気を下押しする状況が継続しております。その他の地域においては、堅調な内需に支えられ景気は回復基調もしくは持ち直しの動きが見られました。わが国経済は、物価上昇や海外経済の回復ペースの鈍化などの影響を受けつつも、経済活動の正常化を背景に内需を中心に緩やかに持ち直しております。</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">　当社グループの当第３四半期連結累計期間における売上高は4,222億94百万円（前年同期比10.3％増）となりました。営業利益は、人件費等の固定費の増加があったものの、原価低減や販売価格の改善などに取り組んだ結果、413億90百万円（前年同期比71.3％増）となりました。経常利益は為替差損や超インフレ会計による正味貨幣持高に係る損失の計上があったものの、持分法投資利益の増加などにより、441億89百万円（前年同期比53.4％増）となりました。親会社株主に帰属する四半期純利益は、政策保有株式縮減に伴う投資有価証券売却益やインドの土地売却に伴う固定資産売却益を計上したことなどにより、539億43百万円（前年同期比221.3％増）となりました。</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph"> </p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">　各セグメントの状況は以下のとおりであります。</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">　なお、第１四半期連結会計期間より、当社グループの経営成績の評価等の検討に使用している主要な経営管理指標を、経常利益から営業利益及び持分法投資損益に変更したことに伴い、セグメント利益も経常利益から営業利益及び持分法投資損益に変更しております。この変更に伴い、前年第３四半期累計期間のセグメント利益も営業利益及び持分法投資損益に変更したうえで比較しております。</p>
<p style="margin-left: 36px; line-height: 13.333333333333333333333333333px; text-align: justify; text-justify: inter-ideograph"> </p>
<p style="margin-left: 36px; line-height: 13.333333333333333333333333333px; text-align: justify; text-justify: inter-ideograph">≪日本≫</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">　自動車分野では自動車生産台数が前年を上回り、売上は前年を上回りました。工業分野、建築分野、自動車分野（補修用）及び防食分野では、市況は低調に推移するものの販売価格の改善に取り組んだことなどからトータルで売上は前年を上回りました。船舶分野では、外航船修繕向けの数量増加などにより売上は前年を上回りました。利益は一部の原材料価格が低下してきたことに加え、販売価格の改善に取り組んだことなどから前年を上回りました。</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">　これらの結果、当セグメントの売上高は1,230億６百万円（前年同期8.7％増）、セグメント利益は164億７百万円（前年同期比99.4％増）となりました。</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph"> </p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">≪インド≫</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">　建築分野では販売促進活動を推進するものの、モンスーンの影響による市況の低迷や競争の激化等の影響を受け、売上は前年並みとなりました。一方、自動車生産は安定しており販売価格の改善も寄与し、インド全体の売上は前年を上回りました。利益は、一部の原材料価格が低下してきたことに加え、販売価格の改善に継続して取り組んだことなどから前年を上回りました。</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">　これらの結果、当セグメントの売上高は1,052億16百万円（前年同期比5.6％増）、セグメント利益は124億69百万円（前年同期比37.6％増）となりました。</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph"> </p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">≪欧州≫</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">　トルコでは、自動車生産台数が前年を上回り、販売価格の改善に取り組んだこともあり、売上は前年を上回りました。その他欧州各国においては、主力の工業分野の売上が堅調に推移したことに加え、販売価格の改善などに取り組んだことにより、売上は前年を上回り、欧州全体としても前年を上回りました。利益はインフレの影響による人件費等のコストの増加があったものの、販売価格の改善に加え一部の原材料価格が低下してきたことなどにより、前年を上回りました。</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">　これらの結果、当セグメントの売上高は1,028億25百万円（前年同期比22.2％増）、セグメント利益は39億53百万円（前年同期比123.6％増）となりました。</p>
<p style="margin-left: 36px; line-height: 13.333333333333333333333333333px; text-align: justify; text-justify: inter-ideograph"> </p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">≪アジア≫</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">　中国においては、自動車生産台数は前年を上回ったものの主要顧客の需要は伸び悩み、売上は前年を下回りました。タイ、マレーシア及びインドネシアにおいては、自動車生産の回復に加え、販売価格の改善の取り組みにより売上は前年を上回りました。利益は一部の原材料価格が低下してきたことに加え、持分法投資利益が増加したことで前年を上回りました。</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">　これらの結果、当セグメントの売上高は534億16百万円（前年同期比6.0％増）、セグメント利益は87億96百万円（前年同期比65.4％増）となりました。</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph"> </p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">≪アフリカ≫</p>
<p style="margin-left: 36px; line-height: 17.333333333333333333333333333px; text-align: justify; text-justify: inter-ideograph">　南アフリカ及び近隣諸国の経済は慢性的な電力不足や物価高などの影響で回復が遅れており需要が低迷するなか、工業分野の需要の取り込みや販売価格の改善などに取り組んだことにより、売上は前年を上回りました。東アフリカ地域においても、建築分野において拡販に注力して売上は堅調に推移し、アフリカ全体の売上は前年を上回りました。利益は安価品原材料への置換などコスト削減に取り組んだことにより、前年を上回りました。</p>
<p style="margin-left: 36px; line-height: 17.333333333333333333333333333px; text-align: justify; text-justify: inter-ideograph">　これらの結果、当セグメントの売上高は310億49百万円（前年同期比1.9％増）、セグメント利益は28億62百万円（前年同期比32.4％増）となりました。</p>
<p style="margin-left: 36px; line-height: 17.6px; text-align: justify; text-justify: inter-ideograph"> </p>
<p style="margin-left: 36px; line-height: 17.6px; text-align: justify; text-justify: inter-ideograph">≪その他≫</p>
<p style="margin-left: 36px; line-height: 17.333333333333333333333333333px; text-align: justify; text-justify: inter-ideograph">　北米では、自動車生産台数が前年を上回り、売上は前年を上回りました。利益については、売上の増加に伴い営業利益が改善したほか、持分法投資利益も増加したことなどにより、前年を上回りました。</p>
<p style="margin-left: 36px; line-height: 17.333333333333333333333333333px; text-align: justify; text-justify: inter-ideograph">　これらの結果、当セグメントの売上高は67億80百万円（前年同期比31.5％増）、セグメント利益は21億66百万円（前年同期比137.9％増）となりました。</p>
<p style="margin-left: 36px; line-height: 17.6px; text-align: justify; text-justify: inter-ideograph"> </p>
<p style="margin-left: 24px; line-height: 17.6px; text-align: justify; text-justify: inter-ideograph">(2）優先的に対処すべき事業上及び財務上の課題</p>
<p style="margin-left: 36px; line-height: 17.6px; text-align: justify; text-justify: inter-ideograph">　当第３四半期連結累計期間において、当社グループが優先的に対処すべき事業上及び財務上の課題について重要な変更はありません。</p>
<p style="margin-left: 36px; line-height: 14.666666666666666666666666667px; text-align: justify; text-justify: inter-ideograph"> </p>
<p style="margin-left: 36px; line-height: 17.6px; text-align: justify; text-justify: inter-ideograph; text-indent: -12px">(3）研究開発活動</p>
<p style="margin-left: 48px; line-height: 17.6px; text-align: justify; text-justify: inter-ideograph; text-indent: -12px">　当第３四半期連結累計期間におけるグループ全体の研究開発活動の総額は、68億59百万円であります。</p>
<p style="margin-left: 48px; line-height: 17.6px; text-align: justify; text-justify: inter-ideograph; text-indent: -12px">　なお、当第３四半期連結累計期間において、当社グループの研究開発活動の状況に重要な変更はありません。</p>
<p style="margin-left: 48px; line-height: 14.666666666666666666666666667px; text-align: justify; text-justify: inter-ideograph; text-indent: -12px"> </p>
<p style="margin-left: 36px; line-height: 17.6px; text-align: justify; text-justify: inter-ideograph; text-indent: -12px">(4）経営成績に重要な影響を与える要因</p>
<p style="margin-left: 48px; line-height: 17.6px; text-align: justify; text-justify: inter-ideograph; text-indent: -12px">　当第３四半期連結累計期間において、経営成績に重要な影響を与える要因について、重要な変更はありません。</p>
<p style="margin-left: 48px; line-height: 14.666666666666666666666666667px; text-align: justify; text-justify: inter-ideograph; text-indent: -12px"> </p>
<p style="margin-left: 24px; line-height: 13.333333333333333333333333333px; text-align: justify; text-justify: inter-ideograph">(5）資本の財源及び資金の流動性についての分析</p>
<p style="margin-left: 48px; line-height: 17.6px; text-align: justify; text-justify: inter-ideograph; text-indent: -12px">（財政状態の状況）</p>
<p style="margin-left: 48px; line-height: 17.6px; text-align: justify; text-justify: inter-ideograph; text-indent: -12px">① 流動資産</p>
<p style="margin-left: 36px; line-height: 17.6px; text-align: justify; text-justify: inter-ideograph">　当第３四半期連結会計期間末における流動資産合計は、3,372億65百万円（前連結会計年度末比174億33百万円増）となりました。流動資産の増加は、現金及び預金などが減少したものの、主に受取手形、売掛金及び契約資産や有価証券などが増加したことによるものであります。</p>
<p style="margin-left: 36px; line-height: 17.6px; text-align: justify; text-justify: inter-ideograph">② 固定資産</p>
<p style="margin-left: 36px; line-height: 17.6px; text-align: justify; text-justify: inter-ideograph">　当第３四半期連結会計期間末における固定資産合計は、3,460億67百万円（前連結会計年度末比60億54百万円減）となりました。固定資産の減少は、有形固定資産や無形固定資産などが増加したものの、投資有価証券などが減少したことによるものであります。</p>
<p style="margin-left: 36px; line-height: 17.6px; text-align: justify; text-justify: inter-ideograph">③ 流動負債</p>
<p style="margin-left: 36px; line-height: 17.6px; text-align: justify; text-justify: inter-ideograph">　当第３四半期連結会計期間末における流動負債合計は、2,342億14百万円（前連結会計年度末比311億17百万円減）となりました。流動負債の減少は、主に短期借入金などが減少したことによるものであります。</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">④ 固定負債</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">　当第３四半期連結会計期間末における固定負債合計は、524億円（前連結会計年度末比12億２百万円減）となりました。</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">⑤ 純資産</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">　当第３四半期連結会計期間末における純資産合計は、3,967億18百万円（前連結会計年度末比436億97百万円増）となりました。</p>
<p style="margin-left: 48px; text-align: justify; text-justify: inter-ideograph; text-indent: -12px"> </p>
<p style="margin-left: 24px; text-align: justify; text-justify: inter-ideograph">(6）経営方針・経営戦略、経営上の目標の達成状況を判断するための客観的な指標等</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">　当社グループは、第３四半期連結累計期間において、経営上の目標の達成状況を判断するための客観的な指標等の見直しを行いました。</p>
<p style="margin-left: 36px; text-align: justify; text-justify: inter-ideograph">　当社グループは、成長性と収益性の両立を図りながら、企業価値の向上を目指しております。第17次中期経営計画の最終年度である2024年度の目標として、連結売上高5,500億円、連結EBITDA850億円、調整後ROE13％超を設定しております。</p>
<p style="margin-left: 24px; text-align: justify; text-justify: inter-ideograph"> </p>
<p style="text-align: left"> </p>





```python
import re


def remove_tags(t):
    return re.compile(r"<.*?>|\n|\t").sub("", t)


text = management_analysis_description.get_text()
remove_tags(text)
```




    '２【経営者による財政状態、経営成績及びキャッシュ・フローの状況の分析】\u3000文中の将来に関する事項は、当四半期連結会計期間の末日現在において判断したものであります。(1）経営成績の状況\u3000当期における世界経済は、供給制約が概ね解消されインフレ率も鈍化の傾向が見られるものの、地政学リスクは依然高まったままで基調的な物価上昇圧力は根強く、欧米を中心に金融引き締めが継続しており、その回復ペースは鈍化しております。そのような状況下、中国においては不動産市況の停滞の影響もありゼロコロナ政策解除後の景気回復は緩慢なペースにとどまっています。欧州においては物価高や利上げによる金融引き締めが景気を下押しする状況が継続しております。その他の地域においては、堅調な内需に支えられ景気は回復基調もしくは持ち直しの動きが見られました。わが国経済は、物価上昇や海外経済の回復ペースの鈍化などの影響を受けつつも、経済活動の正常化を背景に内需を中心に緩やかに持ち直しております。\u3000当社グループの当第３四半期連結累計期間における売上高は4,222億94百万円（前年同期比10.3％増）となりました。営業利益は、人件費等の固定費の増加があったものの、原価低減や販売価格の改善などに取り組んだ結果、413億90百万円（前年同期比71.3％増）となりました。経常利益は為替差損や超インフレ会計による正味貨幣持高に係る損失の計上があったものの、持分法投資利益の増加などにより、441億89百万円（前年同期比53.4％増）となりました。親会社株主に帰属する四半期純利益は、政策保有株式縮減に伴う投資有価証券売却益やインドの土地売却に伴う固定資産売却益を計上したことなどにより、539億43百万円（前年同期比221.3％増）となりました。\xa0\u3000各セグメントの状況は以下のとおりであります。\u3000なお、第１四半期連結会計期間より、当社グループの経営成績の評価等の検討に使用している主要な経営管理指標を、経常利益から営業利益及び持分法投資損益に変更したことに伴い、セグメント利益も経常利益から営業利益及び持分法投資損益に変更しております。この変更に伴い、前年第３四半期累計期間のセグメント利益も営業利益及び持分法投資損益に変更したうえで比較しております。\xa0≪日本≫\u3000自動車分野では自動車生産台数が前年を上回り、売上は前年を上回りました。工業分野、建築分野、自動車分野（補修用）及び防食分野では、市況は低調に推移するものの販売価格の改善に取り組んだことなどからトータルで売上は前年を上回りました。船舶分野では、外航船修繕向けの数量増加などにより売上は前年を上回りました。利益は一部の原材料価格が低下してきたことに加え、販売価格の改善に取り組んだことなどから前年を上回りました。\u3000これらの結果、当セグメントの売上高は1,230億６百万円（前年同期8.7％増）、セグメント利益は164億７百万円（前年同期比99.4％増）となりました。\xa0≪インド≫\u3000建築分野では販売促進活動を推進するものの、モンスーンの影響による市況の低迷や競争の激化等の影響を受け、売上は前年並みとなりました。一方、自動車生産は安定しており販売価格の改善も寄与し、インド全体の売上は前年を上回りました。利益は、一部の原材料価格が低下してきたことに加え、販売価格の改善に継続して取り組んだことなどから前年を上回りました。\u3000これらの結果、当セグメントの売上高は1,052億16百万円（前年同期比5.6％増）、セグメント利益は124億69百万円（前年同期比37.6％増）となりました。\xa0≪欧州≫\u3000トルコでは、自動車生産台数が前年を上回り、販売価格の改善に取り組んだこともあり、売上は前年を上回りました。その他欧州各国においては、主力の工業分野の売上が堅調に推移したことに加え、販売価格の改善などに取り組んだことにより、売上は前年を上回り、欧州全体としても前年を上回りました。利益はインフレの影響による人件費等のコストの増加があったものの、販売価格の改善に加え一部の原材料価格が低下してきたことなどにより、前年を上回りました。\u3000これらの結果、当セグメントの売上高は1,028億25百万円（前年同期比22.2％増）、セグメント利益は39億53百万円（前年同期比123.6％増）となりました。\xa0≪アジア≫\u3000中国においては、自動車生産台数は前年を上回ったものの主要顧客の需要は伸び悩み、売上は前年を下回りました。タイ、マレーシア及びインドネシアにおいては、自動車生産の回復に加え、販売価格の改善の取り組みにより売上は前年を上回りました。利益は一部の原材料価格が低下してきたことに加え、持分法投資利益が増加したことで前年を上回りました。\u3000これらの結果、当セグメントの売上高は534億16百万円（前年同期比6.0％増）、セグメント利益は87億96百万円（前年同期比65.4％増）となりました。\xa0≪アフリカ≫\u3000南アフリカ及び近隣諸国の経済は慢性的な電力不足や物価高などの影響で回復が遅れており需要が低迷するなか、工業分野の需要の取り込みや販売価格の改善などに取り組んだことにより、売上は前年を上回りました。東アフリカ地域においても、建築分野において拡販に注力して売上は堅調に推移し、アフリカ全体の売上は前年を上回りました。利益は安価品原材料への置換などコスト削減に取り組んだことにより、前年を上回りました。\u3000これらの結果、当セグメントの売上高は310億49百万円（前年同期比1.9％増）、セグメント利益は28億62百万円（前年同期比32.4％増）となりました。\xa0≪その他≫\u3000北米では、自動車生産台数が前年を上回り、売上は前年を上回りました。利益については、売上の増加に伴い営業利益が改善したほか、持分法投資利益も増加したことなどにより、前年を上回りました。\u3000これらの結果、当セグメントの売上高は67億80百万円（前年同期比31.5％増）、セグメント利益は21億66百万円（前年同期比137.9％増）となりました。\xa0(2）優先的に対処すべき事業上及び財務上の課題\u3000当第３四半期連結累計期間において、当社グループが優先的に対処すべき事業上及び財務上の課題について重要な変更はありません。\xa0(3）研究開発活動\u3000当第３四半期連結累計期間におけるグループ全体の研究開発活動の総額は、68億59百万円であります。\u3000なお、当第３四半期連結累計期間において、当社グループの研究開発活動の状況に重要な変更はありません。\xa0(4）経営成績に重要な影響を与える要因\u3000当第３四半期連結累計期間において、経営成績に重要な影響を与える要因について、重要な変更はありません。\xa0(5）資本の財源及び資金の流動性についての分析（財政状態の状況）① 流動資産\u3000当第３四半期連結会計期間末における流動資産合計は、3,372億65百万円（前連結会計年度末比174億33百万円増）となりました。流動資産の増加は、現金及び預金などが減少したものの、主に受取手形、売掛金及び契約資産や有価証券などが増加したことによるものであります。② 固定資産\u3000当第３四半期連結会計期間末における固定資産合計は、3,460億67百万円（前連結会計年度末比60億54百万円減）となりました。固定資産の減少は、有形固定資産や無形固定資産などが増加したものの、投資有価証券などが減少したことによるものであります。③ 流動負債\u3000当第３四半期連結会計期間末における流動負債合計は、2,342億14百万円（前連結会計年度末比311億17百万円減）となりました。流動負債の減少は、主に短期借入金などが減少したことによるものであります。④ 固定負債\u3000当第３四半期連結会計期間末における固定負債合計は、524億円（前連結会計年度末比12億２百万円減）となりました。⑤ 純資産\u3000当第３四半期連結会計期間末における純資産合計は、3,967億18百万円（前連結会計年度末比436億97百万円増）となりました。\xa0(6）経営方針・経営戦略、経営上の目標の達成状況を判断するための客観的な指標等\u3000当社グループは、第３四半期連結累計期間において、経営上の目標の達成状況を判断するための客観的な指標等の見直しを行いました。\u3000当社グループは、成長性と収益性の両立を図りながら、企業価値の向上を目指しております。第17次中期経営計画の最終年度である2024年度の目標として、連結売上高5,500億円、連結EBITDA850億円、調整後ROE13％超を設定しております。\xa0\xa0'




```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "あなたは日本企業の決算を分析するアナリストです。"},
        {"role": "user", "content": f"以下の決算のコメントを要約してください。 {remove_tags(text)}"},
        {"role": "user", "content": f"先ほどの決算においてインド市場についてコメントはしていましたか？"},
    ],
)
HTML(response.choices[0].message.content)
```




はい、先ほどの決算においては、インド市場についてもコメントがありました。インド市場では、建築部門においては、モンスーンの影響や競争の激化などにより市況が低迷しているものの、自動車部門では安定した生産と販売価格の改善が売上を前年を上回る水準に押し上げたと述べられています。利益も一部の原材料価格の低下や販売価格の改善により前年を上回っており、インド市場においても一定の成長が見られたことが報告されています。




```python
from IPython.display import Markdown

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "あなたは日本企業の決算を分析するアナリストです。"},
        {"role": "user", "content": f"以下の決算のコメントを要約してください。 {remove_tags(text)}"},
        {"role": "user", "content": f"インドでの業績が好調だった理由を箇条書きで３つ示してください。"},
    ],
)
display(Markdown(response.choices[0].message.content))
```


- 自動車生産が安定しており、販売価格の改善が寄与した
- 一部の原材料価格の低下と販売価格の改善により利益が増加した
- 建築分野では販売促進活動の推進により売上が堅調に推移した



```python
key = "jpcrp_cor:QuarterlyConsolidatedBalanceSheetTextBlock"
context_ref = "CurrentYTDDuration"
balance_sheet = edinet_xbrl_object.get_data_by_context_ref(key, context_ref).get_value()

HTML(balance_sheet)
```





<h4>（１）【四半期連結貸借対照表】</h4>
<div>
<table cellpadding="0" cellspacing="0" style="table-layout: fixed; width: 626.66666666666666666666666667px">
<colgroup>
<col style="width: 309.33333333333333333333333333px"/>
<col style="width: 158.66666666666666666666666667px"/>
<col style="width: 158.66666666666666666666666667px"/>
</colgroup>
<tbody>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 16px; text-align: center">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 16px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 16px; text-align: right">（単位：百万円）</p>
</td>
</tr>
<tr style="height: 37.33px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 3px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 12px; text-align: center">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 3px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 13.333333333333333333333333333px; text-align: center">前連結会計年度</p>
<p style="line-height: 13.333333333333333333333333333px; text-align: center">(2023年３月31日)</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 3px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 13.333333333333333333333333333px; text-align: center">当第３四半期連結会計期間</p>
<p style="line-height: 13.333333333333333333333333333px; text-align: center">(2023年12月31日)</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 10px; line-height: 16px; text-align: left">資産の部</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 27.333333333333333333333333333px; line-height: 16px; text-align: left">流動資産</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">現金及び預金</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
86,973
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
70,211
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">受取手形、売掛金及び契約資産</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
106,785
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 8px">※１</span><span style="font-size: 8px"> </span>129,033
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">有価証券</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
8,169
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
19,116
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">商品及び製品</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
54,673
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
56,624
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">仕掛品</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
7,994
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
8,408
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">原材料及び貯蔵品</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
42,942
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
41,745
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">その他</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
16,819
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
16,827
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">貸倒引当金</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">△4,526</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">△4,701</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">流動資産合計</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
319,832
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
337,265
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 27.333333333333333333333333333px; line-height: 16px; text-align: left">固定資産</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">有形固定資産</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 62px; line-height: 16px; text-align: left">建物及び構築物（純額）</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
65,465
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
68,896
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 62px; line-height: 16px; text-align: left">その他（純額）</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
80,844
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
89,219
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 62px; line-height: 16px; text-align: left">有形固定資産合計</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
146,309
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
158,115
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">無形固定資産</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 62px; line-height: 16px; text-align: left">のれん</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
34,905
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
35,722
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 62px; line-height: 16px; text-align: left">その他</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
28,842
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
32,464
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 62px; line-height: 16px; text-align: left">無形固定資産合計</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
63,747
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
68,186
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">投資その他の資産</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 62px; line-height: 16px; text-align: left">投資有価証券</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
89,098
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
61,482
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 62px; line-height: 16px; text-align: left">その他</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
58,061
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
63,732
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 62px; line-height: 16px; text-align: left">貸倒引当金</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">△5,094</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">△5,450</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 62px; line-height: 16px; text-align: left">投資その他の資産合計</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
142,065
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
119,765
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">固定資産合計</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
352,122
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
346,067
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 309.33333333333333333333333333px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 27.333333333333333333333333333px; line-height: 16px; text-align: left">資産合計</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
671,954
</p>
</td>
<td style="vertical-align: middle; width: 158.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
683,333
</p>
</td>
</tr>
</tbody>
</table>
</div>
<p class="style_pb_after" style="text-align: left"> </p>
<p style="text-align: left"> </p>
<div>
<table cellpadding="0" cellspacing="0" style="table-layout: fixed; width: 626.66666666666666666666666667px">
<colgroup>
<col style="width: 306.66666666666666666666666667px"/>
<col style="width: 160px"/>
<col style="width: 160px"/>
</colgroup>
<tbody>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 16px; text-align: center">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 16px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 16px; text-align: right">（単位：百万円）</p>
</td>
</tr>
<tr style="height: 37.33px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 3px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 12px; text-align: center">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 3px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 13.333333333333333333333333333px; text-align: center">前連結会計年度</p>
<p style="line-height: 13.333333333333333333333333333px; text-align: center">(2023年３月31日)</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 3px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 13.333333333333333333333333333px; text-align: center">当第３四半期連結会計期間</p>
<p style="line-height: 13.333333333333333333333333333px; text-align: center">(2023年12月31日)</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 10px; line-height: 16px; text-align: left">負債の部</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 27.333333333333333333333333333px; line-height: 16px; text-align: left">流動負債</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">支払手形及び買掛金</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
80,999
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 8px">※１</span><span style="font-size: 8px"> </span>89,360
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">短期借入金</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
73,432
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
41,717
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">短期社債</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
44,999
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
45,000
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">未払法人税等</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
7,760
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
16,826
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">賞与引当金</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
4,930
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
4,033
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">その他</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
53,209
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
37,276
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">流動負債合計</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
265,332
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
234,214
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 27.333333333333333333333333333px; line-height: 16px; text-align: left">固定負債</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">退職給付に係る負債</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
7,818
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
8,340
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">その他</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
45,783
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
44,060
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">固定負債合計</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
53,602
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
52,400
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 27.333333333333333333333333333px; line-height: 16px; text-align: left">負債合計</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
318,934
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
286,614
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 10px; line-height: 16px; text-align: left">純資産の部</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 27.333333333333333333333333333px; line-height: 16px; text-align: left">株主資本</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">資本金</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
25,658
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
25,658
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">資本剰余金</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
21,056
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
19,953
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">利益剰余金</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
299,019
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
281,360
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">自己株式</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">△79,971</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">△22,877</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">株主資本合計</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
265,762
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
304,095
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 27.333333333333333333333333333px; line-height: 16px; text-align: left">その他の包括利益累計額</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
<span style="font-size: 12px"> </span>
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">その他有価証券評価差額金</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
32,744
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
15,749
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">繰延ヘッジ損益</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">△2,045</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
7
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">為替換算調整勘定</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">△7,937</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
4,637
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">退職給付に係る調整累計額</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
4,385
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
3,943
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 44.666666666666666666666666667px; line-height: 16px; text-align: left">その他の包括利益累計額合計</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
27,147
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
24,337
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 27.333333333333333333333333333px; line-height: 16px; text-align: left">非支配株主持分</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
60,110
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
68,285
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #CCEEFF">
<p style="margin-left: 27.333333333333333333333333333px; line-height: 16px; text-align: left">純資産合計</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
353,020
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #CCEEFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
396,718
</p>
</td>
</tr>
<tr style="height: 16px">
<td style="vertical-align: middle; width: 306.66666666666666666666666667px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 0px none ; border-right: 0px none ; border-bottom: 0px none ; background-color: #FFFFFF">
<p style="margin-left: 10px; line-height: 16px; text-align: left">負債純資産合計</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
671,954
</p>
</td>
<td style="vertical-align: middle; width: 160px; padding-top: 0px; padding-bottom: 0px; padding-left: 0px; padding-right: 0px; border-left: 0px none ; border-top: 1px solid #000000; border-right: 0px none ; border-bottom: 1px solid #000000; background-color: #FFFFFF">
<p style="line-height: 16px; margin-right: 3px; text-align: right">
683,333
</p>
</td>
</tr>
</tbody>
</table>
</div>





```python
def remove_css(t):
    t = re.compile(r'style=".*?"|\n|\t').sub("", t)
    return re.compile(r"<p >|</p>|<span >|</span>").sub("", t)


response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "あなたは日本企業の決算を分析するアナリストです。"},
        {
            "role": "user",
            "content": f"以下の貸借対照表について、2023年12月31日の純資産合計はいくらとなっていますか。 {remove_css(balance_sheet)}",
        },
    ],
)
display(Markdown(response.choices[0].message.content))
```


2023年12月31日の純資産合計は396,718百万円です。

