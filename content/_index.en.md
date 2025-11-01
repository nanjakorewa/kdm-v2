---
title: "K_DM Book | Data Analysis and Machine Learning Guide"
linkTitle: "Home"
seo_title: "K_DM Book | Data Analysis and Machine Learning Guide"
---
## Welcome to K_DM Book
K_DM Book is a hands-on reference that walks you from the fundamentals of data analysis and machine learning to production-ready workflows. Each section couples narrative guides with runnable notebooks and scripts so you can learn concepts and apply them immediately.

- **Structured learning path** — Content is organized so you can progress from foundations to practical deployment without losing context.
- **Executable examples** — Most articles include Jupyter notebooks or Python scripts that reproduce the figures and experiments.
- **Practical tips** — Evaluation criteria, preprocessing checklists, and visualization patterns are documented with real project insights.

## Suggested learning journey
1. **Basics** — Review core models such as regression, classification, clustering, and dimensionality reduction.
2. **Prep / Evaluation** — Deepen your knowledge of feature engineering and model validation to iterate with confidence.
3. **Timeseries / Finance / WebApp** — Explore domain-specific case studies and understand how to ship models.
4. **Visualize** — Learn how to communicate findings through compelling charts, dashboards, and reports.

## Content map
{{< mermaid >}}
flowchart TD
  subgraph fundamentals["Fundamentals Phase"]
    A1[Mathematical Foundations<br>Linear Algebra / Probability]
    A2[Environment Setup<br>Python & Libraries]
    A3[Exploratory Analysis<br>EDA & Basic Visualization]
  end

  subgraph modeling["Modeling Basics Phase"]
    B1[Regression]
    B2[Classification]
    B3[Clustering]
    B4[Dimensionality Reduction]
    B5[Ensemble Methods]
  end

  subgraph evaluation["Evaluation & Tuning Phase"]
    C1[Metrics]
    C2[Hyperparameter Tuning]
    C3[Feature Engineering]
    C4[Model Explainability]
  end

  subgraph communication["Communication Phase"]
    D1[Visualization Catalog]
    D2[Storytelling / Reporting]
    D3[Collaboration Tips]
  end

  subgraph deployment["Applied & Deployment Phase"]
    E1[Timeseries Analytics]
    E2[Finance Use Cases]
    E3[Web App Delivery<br>Flask / Gradio]
    E4[Monitoring & Operations]
  end

  fundamentals --> modeling
  modeling --> evaluation
  evaluation --> communication
  evaluation --> deployment
  communication --> deployment

  click A2 "/install/" "Go to environment setup"
  click A3 "/prep/" "Explore data preparation guides"
  click B1 "/basic/regression/" "Open the regression section"
  click B2 "/basic/classification/" "Open the classification section"
  click B3 "/basic/clustering/" "Open the clustering section"
  click B4 "/basic/dimensionality_reduction/" "Open the dimensionality reduction section"
  click B5 "/basic/ensemble/" "Open the ensemble section"
  click C1 "/eval/" "Review evaluation metrics"
  click C3 "/prep/feature_selection/" "Go to feature engineering"
  click D1 "/visualize/" "Browse visualization guides"
  click E1 "/timeseries/" "Read timeseries analytics content"
  click E2 "/finance/" "Read finance-focused content"
  click E3 "/webapp/" "Build web applications"
{{< /mermaid >}}

## Supporting resources
- **Notebooks & scripts** — Located under `scripts/` and `data/`, enabling instant reproduction of the tutorials.
- **Updates** — Follow the top-page timeline or subscribe to the RSS feed (`/index.xml`) for new content.
- **Feedback** — Spot an issue? Reach out via the [feedback form](https://kdm.hatenablog.jp/entry/issue) or ping us on X / Twitter.

## Stay connected
- <a href="https://www.youtube.com/@K_DM" style="color:#FF0000;"><i class="fab fa-fw fa-youtube"></i> YouTube — video walkthroughs and live sessions</a>
- <a href="https://twitter.com/_K_DM" style="color:#1DA1F2;"><i class="fab fa-fw fa-twitter"></i> X (Twitter) — quick updates and bite-sized tips</a>

Please review the [privacy policy](https://kdm.hatenablog.jp/privacy-policy) for details about data handling. Enjoy the materials and adapt them to your own projects!
