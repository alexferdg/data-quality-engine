# High-Quality Dataset Research

## Project Overview

This repository hosts research exploring the interplay between data quality, model capabilities, and computational efficiency in artificial intelligence.


## 🚀 Dive Into Synthetic Data Research

[**captioning-data-engine**](https://github.com/alexferdg/data-quality-engine/tree/main/captioning-data-engine)

The `captioning-data-engine` directory hosts an investigation into the generation and utilization of **synthetic data** for improving Vision-Language Models (VLMs).

👀 [Sample Overview](https://alexferdg.github.io/data-quality-engine/)

**Why Synthetic Data?**

- Training models is costly. Quality data is the secret to cutting costs while achieving exceptional performance.
- Learn how **synthetic captions** from VLMs boost datasets like the [1M SBU Captioned Photo Dataset](https://huggingface.co/datasets/vicenteor/sbu_captions).

**How We Do It:**

- Using **state-of-the-art VLMs** like [CogVLM2-llama3-chat-19B](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B) and [InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B) to craft rich, detailed captions.
- Leveraging the [lmdeploy toolkit](https://github.com/InternLM/lmdeploy) to accelerate inference, enabling high-throughput processing of large datasets.
- Validating captions with CLIP-based classification to ensure alignment with image content.

**What We’ve Achieved:**

- **Better Captions:** Significant improvements in alignment scores for synthetic captions over original ones.
- **Smarter Datasets:** Enhanced querying, deduplication, and dataset creation with **vector databases**.
- **Shared Resources:** A [Qdrant database](https://qdrant.tech) of 30,000 synthetic image-caption pairs is made available in the databases folder for further exploration and use.

**Hardware**

All computations were performed on a single `1x A100 SXM4 80GB GPU`.

## Research Questions

### 1. Data Quality and Model Capabilities

- **How does data quality influence the capabilities of machine learning models?**

  > *“Recent work [Sorscher et al., 2022] has shown that it is possible to beat the scaling law using a data curation pipeline”*

### 2. Computational Budget for High-Quality Datasets

- **What are the implications of using high-quality datasets on the computational training budget?**

  > *“When looking at the public* **cloud prices** **for such resources, they are equivalent to hundreds of** **thousands of dollars which is inaccessible to most companies or academic labs**. **But, when using the right ingredients such as having a high-quality dataset** *and leveraging masking strategies when using bigger models,* *training a contrastive model like CLIP on hundreds of millions of images from scratch* *should not require more than 64 GPUs (which should be equivalent to spending* **around 10K USD in compute**). If the VLM that is used for training* *leverages existing pre-trained* *image or text encoder, or LLM,* *the cost of learning a mapping should be much lower.”*  

  Insights from: [An Introduction to Vision-Language Modeling](https://arxiv.org/pdf/2405.17247) by META AI

  

- **How does investing in dataset preparation compare to savings in model training?**

  The computational budget for the dataset can be reused across multiple and varied training runs. Therefore, we hypothesize that in most cases, this budget is fully amortized by subsequent training jobs.

### 3. Model-in-the-Loop "Data Engine"

By incorporating a model-in-the-loop, we aim to create a robust feedback loop where model outputs continually refine data quality, leading to improvements in model performance.


## References

- [The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987) (arXiv)
  > *Summary:* Proposes a theoretical perspective on how large language models learn and represent concepts, bridging abstract theory and practical model behaviour.

- [LAION-POP](https://laion.ai/blog/laion-pop/) (LAION Blog)
  > *Summary:* Discusses LAION’s efforts in curating and sharing massive open datasets for advancing vision-language tasks, emphasising quality and openness.

- [FineWeb Decanting the Web for the Finest Text Data at Scale](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) (Hugging Face)
  > *Summary:* Outlines Hugging Face’s methodology for filtering and refining large-scale web text corpora, demonstrating how more curated data can enhance model performance and efficiency.