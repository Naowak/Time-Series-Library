# EST: Echo State Transformer for Time Series Analysis

This repository contains the implementation and experimental code for **EST (Echo State Transformer)**, a novel architecture that combines Echo State Networks with Transformer mechanisms for time series analysis tasks.

## Overview

EST leverages adaptive working memory through echo state dynamics while maintaining the attention capabilities of Transformers. This hybrid approach demonstrates particular strengths in discriminative tasks such as classification and anomaly detection.

## Repository Structure

This repository is built upon the [Time Series Library (TSLib)](https://github.com/thuml/Time-Series-Library), an open-source benchmark for time series analysis. We have extended TSLib by adding our EST model implementation, allowing for comprehensive evaluation across five mainstream time series tasks:

- **Long-term forecasting**
- **Short-term forecasting** 
- **Imputation**
- **Anomaly detection**
- **Classification**

### Key Additions

- `./models/EST.py` - Complete EST model implementation
- `./scripts/*/EST.sh` - Experiment scripts for all task categories
- `./slurms/` - SLURM batch scripts for distributed computing
- Comprehensive configuration files for reproducible experiments

## Installation

1. Install Python 3.8 and dependencies:

```bash
pip install -r requirements.txt
```

2. Download datasets from [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy) and place them in `./dataset/`

## Usage Examples

### Running EST Experiments

You can reproduce our experimental results using the provided scripts:

```bash
# Anomaly Detection
bash ./scripts/anomaly_detection/MSL/EST.sh
bash ./scripts/anomaly_detection/PSM/EST.sh
bash ./scripts/anomaly_detection/SMAP/EST.sh

# Classification
bash ./scripts/classification/EST.sh

# Long-term Forecasting
bash ./scripts/long_term_forecast/ETT_script/EST_ETTh1.sh
bash ./scripts/long_term_forecast/ETT_script/EST_ETTm1.sh

# Short-term Forecasting
bash ./scripts/short_term_forecast/EST_M4.sh

# Imputation
bash ./scripts/imputation/ETT_script/EST_ETTh1.sh
```

### Using SLURM for Distributed Computing

For large-scale experiments on computing clusters:

```bash
# Submit anomaly detection jobs
sbatch ./slurms/anomaly_detection.sl

# Submit classification jobs  
sbatch ./slurms/classification.sl

# Submit forecasting jobs
sbatch ./slurms/long_term_forecast.sl
```

## Model Configuration

EST supports multiple hyperparameter configurations. Key parameters include:

- `layers`: Number of EST layers
- `memory_units`: Number of memory units in the reservoir
- `memory_dim`: Dimension of memory units
- `model_dim`: Model embedding dimension
- `memory_connectivity`: Sparsity of reservoir connections

All tested configurations are documented in the appendix and available in the repository.

## Reproducibility

This repository provides complete reproducibility of all experimental results presented in our paper. All scripts, configurations, and implementation details are included to ensure exact replication of our findings.

## Baseline Models

The repository includes implementations of state-of-the-art baselines used in our comparison:

- **TimeXer** - TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables [[NeurIPS 2024]](https://arxiv.org/abs/2402.19072)
- **TimeMixer** - TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting [[ICLR 2024]](https://openreview.net/pdf?id=7oLshfEIC2)
- **TSMixer** - TSMixer: An All-MLP Architecture for Time Series Forecasting [[arXiv 2023]](https://arxiv.org/pdf/2303.06053.pdf)
- **iTransformer** - iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [[ICLR 2024]](https://arxiv.org/abs/2310.06625)
- **PatchTST** - A Time Series is Worth 64 Words: Long-term Forecasting with Transformers [[ICLR 2023]](https://openreview.net/pdf?id=Jbdc0vTOcol)
- **Crossformer** - Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=vSVLM2j9eie)
- **Non-stationary Transformer** - Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting [[NeurIPS 2022]](https://openreview.net/pdf?id=ucNDIDRNjjv)
- **TimesNet** - TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis [[ICLR 2023]](https://openreview.net/pdf?id=ju_Uqw384Oq)
- **Mamba** - Mamba: Linear-Time Sequence Modeling with Selective State Spaces [[arXiv 2023]](https://arxiv.org/abs/2312.00752)
- **TiDE** - Long-term Forecasting with TiDE: Time-series Dense Encoder [[arXiv 2023]](https://arxiv.org/pdf/2304.08424.pdf)
- **MICN** - MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting [[ICLR 2023]](https://openreview.net/pdf?id=zt53IDUR1U)
- **DLinear** - Are Transformers Effective for Time Series Forecasting? [[AAAI 2023]](https://arxiv.org/pdf/2205.13504.pdf)
- **LightTS** - Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures [[arXiv 2022]](https://arxiv.org/abs/2207.01186)
- **FreTS** - Frequency-domain MLPs are More Effective Learners in Time Series Forecasting [[NeurIPS 2023]](https://arxiv.org/pdf/2311.06184.pdf)
- **Autoformer** - Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [[NeurIPS 2021]](https://openreview.net/pdf?id=I55UqU-M11y)
- **Informer** - Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [[AAAI 2021]](https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132)
- **FEDformer** - FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [[ICML 2022]](https://proceedings.mlr.press/v162/zhou22g.html)
- **Reformer** - Reformer: The Efficient Transformer [[ICLR 2020]](https://openreview.net/forum?id=rkgNKkHtvB)
- **ETSformer** - ETSformer: Exponential Smoothing Transformers for Time-series Forecasting [[arXiv 2022]](https://arxiv.org/abs/2202.01381)
- **SegRNN** - SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting [[arXiv 2023]](https://arxiv.org/abs/2308.11200.pdf)
- **Koopa** - Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors [[NeurIPS 2023]](https://arxiv.org/pdf/2305.18803.pdf)
- **SCINet** - SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction [[NeurIPS 2022]](https://openreview.net/pdf?id=AyajSjTAzmg)

## Acknowledgments

This work builds upon the excellent [Time Series Library (TSLib)](https://github.com/thuml/Time-Series-Library) developed by Tsinghua University. We thank the TSLib team for providing this comprehensive benchmark platform.

### Citation for TSLib

```bibtex
@article{wang2024tssurvey,
  title={Deep Time Series Models: A Comprehensive Survey and Benchmark},
  author={Yuxuan Wang and Haixu Wu and Jiaxiang Dong and Yong Liu and Mingsheng Long and Jianmin Wang},
  booktitle={arXiv preprint arXiv:2407.13278},
  year={2024},
}
```

## Dataset Sources

All experimental datasets are publicly available and obtained from:

- Long-term Forecasting and Imputation: https://github.com/thuml/Autoformer
- Short-term Forecasting: https://github.com/ServiceNow/N-BEATS  
- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer
- Classification: https://www.timeseriesclassification.com/

## Contact

For questions about the EST model implementation or experimental setup, please open an issue in this repository.

---

**Note**: This is an anonymous repository for paper review purposes. The complete repository with full attribution will be made available upon publication.