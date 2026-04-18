# Enhancing Zero-Shot Time Series Forecasting in Off-the-Shelf LLMs via Noise Injection Prompting
Large Language Models (LLMs) have demonstrated effectiveness as zero-shot time series (TS) forecasters. While existing work often relies on fine-tuning specialized modules to bridge this gap, a distinct, yet challenging, paradigm aims to leverage truly off-the-shelf LLMs without any fine-tuning whatsoever, relying solely on strategic tokenization of numerical sequences.  However, the parameters of these fully frozen models cannot adapt to distribution shifts. Thus, we introduce a novel yet highly effective strategy to overcome this brittleness: injecting noise into the raw time series before tokenization. This non-invasive intervention acts as a form of inference-time augmentation, compelling the frozen LLM to extrapolate based on robust underlying temporal patterns rather than superficial numerical artifacts. We theoretically analyze this phenomenon and empirically validate its effectiveness across diverse benchmarks. Notably, to fully eliminate potential biases from data contamination during LLM pre-training, we introduce multiple novel real-world TS datasets that fall outside all utilized LLMs' pre-training scopes, and consistently observe improved performance. This study provides a further step in directly leveraging off-the-shelf LLMs for time series forecasting.

![Overview of zero-shot TS forecasting in off-the-shelf LLMs: the top is a vanilla usage of off-the-shelf LLM for TS, where the numerical values are tokenized and directly converted into a string, and then fed into a frozen LLM for prediction. The bottom is our NLTS framework, which introduces noise injection.](overview_01.png)
# Citation
```bibtex
@misc{yin2025enhancingzeroshottimeseries,
      title         = {Enhancing Zero-Shot Time Series Forecasting in Off-the-Shelf LLMs via Noise Injection}, 
      author        = {Xingyou Yin and Ceyao Zhang and Min Hu and Kai Chen},
      year          = {2025},
      eprint        = {2512.20140},
      archivePrefix = {arXiv},
      primaryClass  = {cs.AI},
      url           = {https://arxiv.org/abs/2512.20140}, 
}
```
