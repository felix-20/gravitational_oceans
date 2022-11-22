# Progress in the gravitational ocean

## 25-Oct-2022
- understand the challenge
- see PPP

## 01-Nov-2022
- setup repo
- first tests with efficentnet

## 08-Nov-2022
- Experimented with fastAI
- play with signal-to-noise-ratio:
    - high ratio: good signals -> AI learns and predicts well
    - low ratio: little signal -> not get better than guessing (0.5)
- problems with Charlie
- looked at papers ([paper1](https://arxiv.org/ftp/arxiv/papers/1904/1904.13291.pdf), [paper2](https://arxiv.org/pdf/1908.11170.pdf))
- started visualizing data

## 15-Nov-2022
- first submission: cheated the challenge, no AI, just interpolation (see cheat-task notebook kaggle)
- started crnn from [this link](https://github.com/dredwardhyde/crnn-ctc-loss-pytorch)
- created new dataset from kaggle training data
- our generated data is not really suitable for training, because it has 5000 frequencies, but real data only has 360

## 22-Nov-2022
- fixed problems with CRNN, works now
- set fixed sequence length for crnn, it does not need to guess
- use tensorboard to visualize training progress
- read [this paper](https://arxiv.org/pdf/2203.06717.pdf) -> increase kernel size
- setup data and code on charlie -> took a while
