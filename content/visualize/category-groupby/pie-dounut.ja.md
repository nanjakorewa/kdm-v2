---
title: "ドーナツチャート"
pre: "6.1.3 "
weight: 3
not_use_colab: true
searchtitle: "pythonでドーナツチャートを作成する"
---

ドーナツチャート（ドーナツグラフ）はカテゴリごとの比率を表示するために使用される円グラフの一種で、真ん中に空白があります。空白に特別な意味はありませんが、ここに全体の統計量（たとえば、『合計XXX円』など）を表示したりします。ドーナツチャートは[matplotlib.pyplot.pie](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html)を使ってpython上でドーナツチャートを作成します。


```python
import matplotlib.pyplot as plt

# プロット用のデータ
percent = [40, 20, 20, 10, 10]
explode = [0, 0, 0, 0, 0]
labels = ["米国", "エマージング", "日本", "欧州", "その他"]

percent.reverse()
explode.reverse()
labels.reverse()

# 円グラフを作成
plt.figure(figsize=(7, 7))
plt.pie(x=percent, labels=labels, explode=explode, autopct="%1.0f%%", startangle=90)

# 真ん中に空白の円を追加
background_color = "#fff"
p = plt.gcf()
p.gca().add_artist(plt.Circle((0, 0), 0.8, color=background_color))

plt.show()
```


    
![png](/images/visualize/category-groupby/pie-dounut_files/pie-dounut_1_0.png)
    

