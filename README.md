# MiFGD

## Overview

This repository provides code implementation of "Fast quantum state reconstruction via accelerated non-convex programming" (https://arxiv.org/abs/2104.07006). The proposed method is called Momentum-Inspired Factored Gradient Descent (MiFGD), and it combines ideas from compressed sensing, non-convex optimization, and acceleration methods. In a nutshell, MiFGD  can reconstruct an unknown low-rank density matrix given the corresponding set of measurements much more efficiently compared to other convex methods.

## Install

- Install Miniconda: https://docs.conda.io/en/latest/miniconda.html

- Create a conda environment: `mifgd`
```
conda env create -f environment.yml
```

## Run the example notebook

- Activate the environment:
```
conda activate mifgd
```

- `cd` to  `notebooks/`:
```
cd notebooks
```

- Launch a `jupyter lab` session from a terminal:
```
jupyter lab
```

- Work in your browser: Open `qutomo_example.ipynb` and execute its cells.


- When you finish working, deactivate the environment: 

```
conda deactivate
```

## Citation

```
@article{kim2021fast,
  title={Fast quantum state reconstruction via accelerated non-convex programming},
  author={Kim, Junhyung Lyle and Kollias, George and Kalev, Amir and Wei, Ken X and Kyrillidis, Anastasios},
  journal={arXiv preprint arXiv:2104.07006},
  year={2021}
}
```

