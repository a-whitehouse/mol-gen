# Config

## Preprocessing

The preprocessing config yml file should have the following structure:

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
    structure_filters:
        - smiles: <SMILES string a>
          min: <numeric>
        - smiles: <SMILES string b>
          min: <numeric>
split:
    validate: <numeric>
    test: <numeric>
```

An example config file can be found [here](examples/preprocessing.yml).

### convert

Here you can specify which conversion methods should be executed by the preprocessing step.

Example:
```yaml
convert:
    - neutralise_salts
    - remove_fragments
    - remove_isotopes
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

**structure_filters**

Here you can specify the structures the molecules should all share.
A SMILES string for the structural element and a minimum Tanimoto similarity score should be specified.

Example:
```yaml
    structure_filters:
        - smiles: CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O
          min: 0.2
```

Similarity is calculated by converting the SMILES string and the molecule being compared to Morgan fingerprints.
The minimum Tanimoto similarity score should be a number in the interval (0, 1].


**split**

Here you can specify the sizes of the validation and test splits.

Example:
```yaml
split:
    validate: 0.1
    test: 0.1
```

The size of the train split will be determined automatically from the remainder of the validation and test splits.
Hence, the total size of the validation and test splits should be less than 1.


## Training

The training config yml file should have the following structure:

```yaml
dataset:
    buffer_size: <numeric>
    batch_size: <numeric>
model:
    embedding_dim: <numeric>
    lstm_layers:
        - units: <numeric>
          dropout: <numeric>
        - units: <numeric>
          dropout: <numeric>
    patience: <numeric>
    epochs: <numeric>
evaluate:
    n_molecules: <numeric>
    subset_size: <numeric>
```

**dataset**

Here you can configure the Tensorflow [input pipeline](https://www.tensorflow.org/guide/data).


Example:
```yaml
dataset:
    buffer_size: 1000000
    batch_size: 2048
```

The [shuffling](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle) of the dataset is configured by *buffer_size*.
The size of the [padded batch](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#padded_batch) is configured by *batch_size*.

**model**

Here you can configure the model architecture and the number of epochs to train (optional, defaults to 100).

Example:
```yaml
model:
    embedding_dim: 64
    lstm_layers:
        - units: 256
          dropout: 0.3
        - units: 256
          dropout: 0.5
    patience: 5
    epochs: 100
```

The first layer is configured by *embedding_dim*, the size of the vector the input integers are transformed into.

The subsequent LSTM layers are configured by *lstm_layers*,
where you specify *units* and *dropout* (optional, defaults to 0).
The above example configures 2 LSTM layers, but more can be specified.

Early stopping is specified by *patience* (optional, defaults to 5).

**evaluate**

Here you can configure the model evaluation report.

Example:
```yaml
evaluate:
    n_molecules: 1024
    subset_size: 50
```

The number of molecules to generate is configured by *n_molecules*.
The number of molecules to draw is configured by *subset_size*,
which must be less than the number of molecules generated.
