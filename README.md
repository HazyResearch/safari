# Convolutions for Sequence Modeling

This repository provides implementations and experiments for the following papers, as well as simplified presentations of earlier work such as [S4](https://github.com/HazyResearch/state-spaces).
This repository will eventually merge in code from H3 and LLM training scripts.

## Long Convs

**Simple Hardware-Efficient Long Convolutions for Sequence Modeling**\
Daniel Y. Fu*, Elliot L. Epstein*, Eric Nguyen, Armin W. Thomas, Michael Zhang, Tri Dao, Atri Rudra, Christopher Ré\
[Paper](https://arxiv.org/abs/2302.06646)
![LongConvs](assets/long_convs.png "Long Convolutions for Sequence Modeling")

### Roadmap
- Include H3, LLM training, and synthetics in this repository
- Move in fast convolution code
- pip package

### Changelog
See [CHANGELOG.md](CHANGELOG.md)

## Setup

### Requirements
This repository requires Python 3.8+ and Pytorch 1.10+.
Other packages are listed in [requirements.txt](./requirements.txt).

## Getting Started
The easiest way to get started is to run the [`standalone_cifar.py`](./standalone_cifar.py) script.
This scripts trains a simple long convolution model on CIFAR-10:
```
python -m standalone_cifar
```

To reproduce LRA and CIFAR experiments from the paper, see the [experiments](./experiments.md) page.

## Citation

If you use this codebase, or otherwise found our work valuable, you can cite us as follows:
```
@article{fu2023simple,
  title={Simple Hardware-Efficient Long Convolutions for Sequence Modeling},
  author={Fu, Daniel Y. and Epstein, Elliot L. and Nguyen, Eric and Thomas, Armin W. and Zhang, Michael and Dao, Tri and Rudra, Atri and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:2302.06646},
  year={2023}
}
```

## Acknowledgements

This repo was forked from Albert Gu's [state spaces](https://github.com/HazyResearch/state-spaces) repo and borrows its structure.
It also contains code from the [FlashAttention](https://github.com/HazyResearch/flash-attention) training scripts.