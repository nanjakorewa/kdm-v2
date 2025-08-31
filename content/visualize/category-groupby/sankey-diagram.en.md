---
title: "Sankey Diagram"
pre: "6.1.4 "
weight: 4
not_use_colab: true
searchtitle: "Creating a Sankey Diagram in python"
---

Sankey diagrams are charts that represent the flow rate between steps and can be utilized to visualize how the allocation of amounts and resources is changing. The thickness of the line indicates the amount of flow allocated. This page uses plotly to create a sankey diagram in python.

Please note that the following is a visualization of profit and loss for one period for "[Genco Shipping & Trading Limited](https://www.gencoshipping.com/)" and **not the latest data**.

{{% notice document %}}
[Sankey Diagram in Python | plotly](https://plotly.com/python/sankey-diagram/)
{{% /notice %}}


```python
import plotly.graph_objects as go

fig = go.Figure(
    data=[
        go.Sankey(
            node=dict(
                pad=300,
                thickness=20,
                line=dict(color="white", width=0.0),
                label=[
                    "Voyage revenues",
                    "Total revenues",
                    "Total operating expenses",
                    "Net income",
                    "Voyage expenses",
                    "Vessel operating expenses",
                    "Charter hire expenses",
                    "General and administrative expenses",
                    "Depreciation and amortization",
                ],
                color=[
                    "#666666",
                    "#666666",
                    "#CC0001",
                    "#2BA02D",
                    "#CC0001",
                    "#CC0001",
                    "#CC0001",
                    "#CC0001",
                    "#CC0001",
                ],
            ),
            link=dict(
                source=[0, 1, 1, 2, 2, 2, 2, 2],
                target=[1, 2, 3, 4, 5, 6, 7, 8],
                value=[121008, 84759, 36249, 36702, 18789, 8325, 5854, 13769],
                color=[
                    "#B3B3B3",
                    "#E18685",
                    "#9CCC9A",
                    "#E18685",
                    "#E18685",
                    "#E18685",
                    "#E18685",
                    "#E18685",
                ],
            ),
        )
    ]
)

fig.update_layout(
    title_text="Condensed Consolidated Statements of Operations", font_size=18
)
fig.show()
```

![png](/images/visualize/category-groupby/sankey-diagram.png)