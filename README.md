# Molecule Generation

## Author

Andrew Whitehouse

## Background

This project is inspired by the 2018 research paper [Generative Recurrent Networks for De Novo Drug Design](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5836943/).
It is quite readable, and gives a nice demonstration of the use of neural networks with [LSTM layers](https://en.wikipedia.org/wiki/Long_short-term_memory) to generate novel molecules after training on a curated [ChEMBL](https://www.ebi.ac.uk/chembl/) dataset.
I also found the following 2021 review [De novo molecular design and generative models](https://www.sciencedirect.com/science/article/pii/S1359644621002531) helpful in gaining a broader understanding of the field.

Another tool I wanted to experiment with is the use of Self-Referencing Embedded Strings (SELFIES) in place of SMILES strings for the encoding of molecules.
SELFIES are a [recent development](https://iopscience.iop.org/article/10.1088/2632-2153/aba947) that allow molecules to be represented in a language that is easier for computers,
avoiding the learning of complex syntax that is required for SMILES strings.
Whilst it is possible to have a SMILES string that cannot be parsed due to incorrect grammar,
it is much more difficult to do so with SELFIES.
Hence, using SELFIES for training recurrent neural networks should afford better results,
as training will be able to focus more on the specifics of molecular structure rather than syntax.

This new project is my naive exploration on the use of [recurrent neural networks (RNNs)](https://en.wikipedia.org/wiki/Recurrent_neural_network) for *de novo* molecule generation using SELFIES.
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

### Hardware

For preprocessing I parallelised the application of the [MoleculePreprocessor](src/mol_gen/preprocessing/preprocessor.py) to the input parquet dataframe through a Dask [distributed scheduler](src/mol_gen/preprocessing/dask.py).
This was able to provide a 4-fold speedup on my 6-core Ryzen 5600X CPU.
The use of Dask also allows the preprocessing step to handle larger-then-memory datasets, which was very useful for when my PC only had 16 GB RAM.

For model training I made use of my PC's GPU, a GeForce RTX 3080.
The libraries Cudnn, Cudatoolkit and tensorflow-gpu in the environment allow Tensorflow to do this without extra effort.
Training will be significantly slower if you attempt training without a CUDA-enabled GPU.


### Dataset

The project requires a dataset of molecules to preprocess and train.
The molecules should be provided initially as SMILES strings, stored within a column of a parquet file/ directory.
The small test dataset [chembl.parquet](tests/data/chembl.parquet) is provided for demonstration purposes.
Actual datasets though should be much larger.
I personally downloaded a large [dataset](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz) of ~110 million molecules from [PubChem](https://pubchem.ncbi.nlm.nih.gov/).

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
It is here that the SMILES strings are translated into SELFIES for training.
The step is designed to be completely configurable so you can train the neural networks on multiple datasets, each with different properties.
To achieve this, the preprocessing step requires a config file.
The structure of the preprocessing config file is described in the [config docs](docs/config.md#preprocessing).

A typical workflow would be to neutralise salts, remove isotopic and stereochemical information,
and filter molecules that are not [druglike](https://en.wikipedia.org/wiki/Druglikeness).
The resultant dataset of SMILES strings would then be converted to SELFIES,
with 80% kept in the training set and 10% reserved for validation and testing respectively.
This can all be achieved with the example preprocessing [config file](examples/all_drug_like/preprocessing.yml).

The file can be made more restrictive if desired, for example to only keep molecules that have a minimum Tanimoto similarity score to a particular lead compound from a high-throughput screening campaign.
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

## Examples

### All Drug-Like

The "all drug-like" example directory shows the outcome from training on a dataset of 60 million molecules.
This dataset was obtained from the original PubChem dataset as [described previously](#preprocess) using Lipinski's rules with no structural similarity filtering.


## Further Work

I have only just got to the stage where the report generation is successfully integrated with Tensorflow's callbacks system.
The next step would be to focus more on improving the content and overall presentation of the reports.

The customisation of the model architecture is currently limited,
as a lot of my attention was focused on getting the preprocessing and molecule generation steps to work.
The model only has one recurrent layer, so it would be interesting to explore more complex models.

I need unit tests in the model evaluation and molecule generation sections of the codebase.
