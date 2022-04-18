# Molecule Generation

## Author

Andrew Whitehouse

## Background

This project is inspired by the 2018 research paper [Generative Recurrent Networks for De Novo Drug Design](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5836943/) by Gupta *et al*.
It is quite readable, and gives a nice demonstration of the use of neural networks with [LSTM layers](https://en.wikipedia.org/wiki/Long_short-term_memory) to generate novel molecules after training on a curated [ChEMBL](https://www.ebi.ac.uk/chembl/) dataset.
I also found the following 2021 review [De novo molecular design and generative models](https://www.sciencedirect.com/science/article/pii/S1359644621002531) by Meyers *et al.* helpful in gaining a broader understanding of the field.

Whilst I have a background in chemistry, it is in synthetic lab-based medicinal chemistry rather than hardcore cheminformatics.
I first learned Python in 2018 for fun during my PhD on [fragment-based methods](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00809), and one of the first test projects I made was on the design of fragments from scratch, AKA *de novo* molecule generation.
You would specify a molecular formula, e.g. C<sub>4</sub>NO, and the program would recursively generate all possible molecules based on rules that I hardcoded.
I quickly ran into the problem of exponential growth, and many of the molecules generated were chemically implausible.
After a few weeks, I put this project on indefinite hold and focused my efforts elsewhere.

This new project is my naive exploration on the use of [recurrent neural networks (RNNs)](https://en.wikipedia.org/wiki/Recurrent_neural_network) for *de novo* molecule generation, as in the paper by Gupta *et al.* and many other papers that I haven't had the time to read.
By doing so, I hope to gain a better understanding of RNNs, LSTMs, variational autoencoders, and cheminformatics in general.

## Project Structure

This project is intended to have 3 distinct sections:
1. Preprocessing of registered molecules to create curated datasets
2. Training of neural networks on these datasets
3. Analysis and visualisation on the output of the trained models

## Environment Setup

TODO

## Preprocessing

TODO

## Training

TODO

## Analysis

TODO
