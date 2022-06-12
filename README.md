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
I will definitely retread ground that was covered by previous work, but I have to start somewhere.

## Project Structure

The project is divided into 3 distinct sections:
1. Preprocessing of molecules to create curated datasets.
2. Training of neural networks on these datasets.
3. Generation of molecules from the trained models.

## Prerequisites

### Environment Setup

The project is based on Python 3.10, with the environment managed by [Anaconda](https://www.anaconda.com/) and run in Windows.
The environment can be recreated from the [environment.yml](environment.yml) file by:

```
conda env create -f environment.yml
conda activate molGenEnv
```

### Dataset

The project requires a dataset of molecules to preprocess and train.
The molecules should be provided as SMILES strings, stored within a column of a parquet file/ directory.
The small test dataset [chembl.parquet](tests/data/chembl.parquet) is provided for demonstration purposes.
Actual datasets though should be much larger.
I personally downloaded a large [dataset](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz) of molecules from [PubChem](https://pubchem.ncbi.nlm.nih.gov/).

## Usage

With the environment activated as [described previously](#environment-setup),
you should have access to the `mol-gen` entry point.
The available commands (preprocess, train and generate) can be viewed by:

```
mol-gen --help
```

Arguments for each command can be viewed similarly.
For example:

```
mol-gen preprocess --help
```

### Preprocess

The preprocessing step allows you to convert, filter and split a dataset of molecules prior to training.
The SMILES strings are also translated into SELFIES for training.
The step is designed to be completely configurable so you can train the neural networks on multiple datasets, each with different properties.
To achieve this, the preprocessing step requires a config file.
The structure of the preprocessing config file is described in the [config docs](docs/config.md#preprocessing).

A typical workflow would be to neutralise salts, remove isotopic and stereochemical information,
and filter molecules that are not [druglike](https://en.wikipedia.org/wiki/Druglikeness).
The resultant dataset of SMILES strings would then be converted to SELFIES,
with 80% kept in the training set and 10% reserved for validation and testing respectively.
This can all be achieved with the example preprocessing [config file](examples/all_drug_like/preprocessing.yml).

The file can be made more restrictive if desired, for example to only keep molecules that have a Tanimoto similarity score of at least 0.5 to a particular lead compound from a high-throughput screening campaign.
To achieve this, you can expand the config to include structure filters as described in the [config docs](docs/config.md#preprocessing).

Once you have a config file you can run the preprocessing step on a dataset of SMILES strings:

```
mol-gen preprocess --config <path to config file> --input <path to dataset> --column <name of column in dataset> --output <path to output directory>
```

For example:

```
mol-gen preprocess --config examples\all_drug_like\preprocessing.yml --input tests\data\chembl.parquet --column SMILES --output examples\test_run\preprocessed
```

The preprocessing step will create the following directory structure in the output directory:

```
<output directory>
├── selfies
│   ├── test
│   │   ├── 0.part
│   │   ├── 1.part
│   │   ├── 2.part
│   │   └── ...
│   ├── train
│   │   ├── 0.part
│   │   ├── 1.part
│   │   ├── 2.part
│   │   └── ...
│   ├── validate
│   │   ├── 0.part
│   │   ├── 1.part
│   │   ├── 2.part
│   │   └── ...
│   └── token_counts.csv
└── smiles
    ├── part.0.parquet
    ├── part.1.parquet
    ├── part.2.parquet
    └── ...
```

The smiles directory contains the converted and filtered SMILES strings stored in parquet format.
The selfies directory contains the same molecules represented as SELFIES and further split into train, validate and test splits.
Each split is stored across multiple text files.
A token counts file is also provided, showing the total counts for each SELFIES token across all splits.

### Train

The training step allows you to train a recurrent neural network on the preprocessed SELFIES.
It is designed around Tensorflow, and executes the following:
1. Setup of an [input pipeline](https://www.tensorflow.org/guide/data) from the preprocessed SELFIES dataset.
2. Compiling of a Keras model with callbacks.
3. Fitting of the model.

The input pipeline shuffles the dataset, pads the SELFIES with start- and end-of-sequence characters,
then tokenises the SELFIES using the [token counts](C:\Users\ajw37\Documents\mol-gen\tests\data\trained\string_lookup.json) to construct a vocabulary.
The tokenised molecules are organised into batches of the same length.
The molecules are finally split to input and target sequences that are shifted by one character relative to each other.

As with preprocess, training is designed to be configurable so you can train the neural networks with different hyperparameters.
To achieve this, the training step requires a config file.
The structure of the training config file is described in the [config docs](docs/config.md#training).

Once you have a config file you can run the training step on a preprocessed dataset of SELFIES:

```
mol-gen train --config <path to config file> --input <path to preprocessed dataset> --output <path to output directory>
```

For example:

```
mol-gen train --config examples\all_drug_like\training.yml --input examples\test_run\preprocessed --output examples\test_run\trained
```

The training step will create the following directory structure in the output directory:

```
<output directory>
├── checkpoints
│   ├── model.01.h5
│   ├── model.02.h5
│   ├── model.03.h5
│   └── ...
├── logs
├── reports
│   ├── model.01.html
│   ├── model.02.html
│   ├── model.03.html
│   └── ...
└── string_lookup.json
```

The checkpoints are versions of the model saved after each epoch,
and can be [loaded by Tensorflow](https://www.tensorflow.org/tutorials/keras/save_and_load).
HTML reports are also generated for each checkpoint,
allowing you to monitor the performance of each checkpoint model.

The string lookup JSON is the config required to map SELFIES to the tokens used by the models.

### Generate

The molecule generation step allows you to easily generate molecules from the trained model checkpoints.

It takes the string lookup JSON and a model checkpoint created by the train step:

```
mol-gen generate --model <path to trained model> --vocab <path to string lookup JSON> --output <path to output file> --n-mols <number of molecules to generate>
```

For example:
```
mol-gen generate --model examples\all_drug_like\trained\checkpoints\model.13.h5 --vocab examples\all_drug_like\trained\string_lookup.json --output data\test_run --n-mols 100
```

The generate step will create a text file containing SELFIES:

```
[=Branch1][C][=N][C][C][=N][NH1][N][=Ring1][Branch1]
[Cl+1][=N+1][=N][C][=C][NH1][N][=C][C][Ring1][=Branch1][=S]
[\O-1][=C][Branch1][Ring1][C][N][C][C][N][C][Branch1][C][C][=O]
[/NH1-1][=C][Branch1][C][O][C][C][C][C][Branch1][C][Cl][C][Ring1][#Branch1]
[=N][#C][=N][C][C][=C][Branch1][Branch1][C][C][C][O][O][Ring1][=Branch2]
[=Branch1][O][C][=C][N][=C][C][=C][C][=C][C][Ring1][=Branch1][=N][Ring1][#Branch2]
[#SH2][=C][C][=C][C][=C][C][=C][C][=C][C][=C][Ring1][#Branch2][Ring1][=Branch1]
[C][C][=Branch1][C][=O][N][C][=C][Branch1][C][O][C][=C][C][=N][Ring1][#Branch1]
[SH1+1][=N+1][=Branch1][Branch2][=C][C][=C][C][C][N][C][C][C][C][C][Ring1][=Branch1]
[#C][=C][C][=C][C][Branch1][=Branch2][C][=N][C][=N][N][Ring1][Branch1][C][=N][Ring1][N]
```
