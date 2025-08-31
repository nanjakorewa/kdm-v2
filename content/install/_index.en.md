---
title: Introduction
weight: 1
pre: "<b>1. </b>"
chapter: true
not_use_colab: true
not_use_twitter: true
---

### Section 1
# What is this page?

This site is a page where _k (site creator) leaves a vlog and jupyter notebook of what he studied.


## How to run code

{{< tabs >}}
{{% tab name="Google Colaboratory(Colab)" %}}
Google Colaboratory(**Colab**) is the name of the Python runtime environment provided by Google that runs in the browser. Colab allows you to easily execute and share code without building any virtual environment.
A Google account is required to use this service.

Click on the<a href="#" class="btn colab-btn-border in-text-button">Colab</a> button to run the code on that page on Google Colaboratory.

![](2022-03-30-20-15-24.png)

{{% /tab %}}
{{% tab name="Anaconda" %}}

After creating a virtual environment with Anaconda, install the necessary library packages for each notebook and code. Please note, however, this site does not use Anaconda at all and has not been tested.

```
conda create -n py38_env python=3.8
conda install *library_name*
```

{{% /tab %}}
{{% tab name="Poetry" %}}

After installing POETRY in the manner described [Poetry | Installation](https://python-poetry.org/docs/),
run `poetry install` command.

{{% /tab %}}
{{< /tabs >}}