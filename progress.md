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

## 29-Nov-2022
- analyzed kaggle solution discussed last week
- adding more noise is valid augmentation (maybe even the best augemtation)
- setup and successfully develop stuff on charlie (using ssh, git repo)

## 08-Dec-2022
- prepared presentation for last week Thursday
- started with transformer from [this article](https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1)
- understood template code and modified to work with the trained CNN data
- adjusted dataloader
- train 2/3 different transformers on CNN output (33 h of training)
- with transformer and seq_len 16 we are way better than guessing, accuracy around 62 %
