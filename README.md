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

This project is intended to have 3 distinct sections:
1. Preprocessing of registered molecules to create curated datasets
2. Training of neural networks on these datasets
3. Analysis and visualisation on the output of the trained models

## Environment Setup

I am using an environment with Python 3.9 managed by [Anaconda](https://www.anaconda.com/) and run in Windows.
To recreate it from my [environment.yml](environment.yml) file, use the following in the command prompt:

```
conda env create -f environment.yml
conda activate molGenEnv
```

## Development

I am using [Visual Studio Code](https://code.visualstudio.com/) for my development work.
You can find the settings I used in my workspace [here](.vscode/settings.json), including the use of the linters [black](https://black.readthedocs.io/en/stable/) and [flake8](https://flake8.pycqa.org/en/latest/).

I use the [pre-commit](https://pre-commit.com/) library to enforce code quality, with my specific git hooks described [here](.pre-commit-config.yaml).

## Preprocessing

I have downloaded a large [dataset](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz) of molecules from [PubChem](https://pubchem.ncbi.nlm.nih.gov/) for this project.
Before training I want to preprocess this dataset to neutralise salts, remove stereochemical information, and filter molecules that are not [druglike](https://en.wikipedia.org/wiki/Druglikeness).
The preprocessing step should be configurable so I can train the neural networks on multiple datasets, each with different properties.
To achieve this, the preprocessing step requires a config file.

To run the preprocessing step, execute the [run_preprocessing.py](scripts/run_preprocessing.py) script:
```
python scripts/run_preprocessing.py --config <path to config file> --input <path to directory containing full dataset of molecules> --output <path to directory to write preprocessed dataset>
```

The input directory should contain csv files with a "SMILES" column containing SMILES strings of molecules.

The config yml file should have the following structure:
```yaml
convert:
    - <convert method a>
    - <convert method b>
filter:
    allowed_elements: [H, C, N, ...]
    range_filters:
        <descriptor a>:
            min: <numeric>
            max: <numeric>
        <descriptor b>:
            min: <numeric>
            max: <numeric>
```

An example config file can be found [here](examples/preprocessing.yml).

### convert

Here you can specify which conversion methods should be executed by the preprocessing step.

Example:
```yaml
convert:
    - neutralise_salts
    - remove_stereochemistry
```

### filter

**allowed_elements**

Here you can specify the elements that molecules are allowed to have.
If a molecule has an element that is not in this list, it will be filtered out.

Example:
```yaml
    allowed_elements: [H, C, N, O, F, S, Cl, Br]
```

**range_filters**

Here you can specify the allowed ranges of values for descriptors that molecules can have.
Either a min or max value, or both, should be specified.

Example:
```yaml
    range_filters:
        hydrogen_bond_acceptors:
            max: 10
        hydrogen_bond_donors:
            max: 5
        molar_refractivity:
            min: 40
            max: 130
        molecular_weight:
            min: 180
            max: 480
        partition_coefficient:
            min: -0.4
            max: 5.6
        rotatable_bonds:
            max: 10
        topological_polar_surface_area:
            max: 140
```

## Training

TODO

## Analysis

TODO
