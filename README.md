# bit_collision_free_MF

A Python package for generating molecular fingerprints without bit collisions.

## Description

`bit_collision_free_MF` generates **count-based** Morgan fingerprints while eliminating bit collisions, which can significantly improve the accuracy and reliability of molecular fingerprints in cheminformatics applications.

Two modes are supported:

- **Hashed mode** (default): Uses RDKit's hashed Morgan fingerprint and automatically finds the minimal fingerprint length that avoids all bit collisions via exponential growth + binary search. Collision detection is based on comparing Morgan **invariants** (unique substructure identifiers), which catches all collision types — including collisions between different substructures at the same radius.
- **Unfolded mode**: Builds a direct mapping from Morgan invariant IDs to column indices. Each unique substructure gets its own column, guaranteeing zero collisions with the **minimal possible number of columns**. This mode is especially useful for large datasets where the hashed mode would produce very long fingerprints, and it provides full interpretability via `get_invariant_mapping()`.

## Installation

### Requirements

- Python 3.9 or higher
- numpy
- pandas
- rdkit

### Simple Installation

```bash
pip install -U bit_collision_free_MF
```

This will automatically install all dependencies, including RDKit.

### Manual Installation

```bash
# Install dependencies
pip install numpy pandas rdkit

# Install the package
pip install -U bit_collision_free_MF
```

For development installation:
```bash
git clone https://github.com/Shifa-Zhong/bit_collision_free_MF.git
cd bit_collision_free_MF
pip install -e .
```

## Features

- **Two modes**: `"hashed"` (optimized-length hashed fingerprint) and `"unfolded"` (direct invariant-to-column mapping)
- **Invariant-based collision detection**: compares actual substructure identifiers, not just radii, to guarantee truly collision-free fingerprints
- **Count-based output**: generates count fingerprints (not binary), preserving substructure frequency information
- **Unfolded mode benefits**: minimal column count (= number of unique substructures), lower memory usage for large datasets, and full interpretability via `get_invariant_mapping()`
- Supports all radius values including radius=0
- **Consistent zero-column removal**: columns identified during `fit()` are reused in `transform()`, ensuring train/test dimensionality alignment
- Feature names (`fp_0`, `fp_1`, ...) are 0-indexed to match bit positions in RDKit's `bitInfo`, enabling correct substructure interpretation
- Easy CSV export with customizable headers
- Seamless integration with pandas and NumPy

## Usage

### Basic Usage (Hashed Mode — default)

```python
from bit_collision_free_MF import generate_fingerprints, save_fingerprints
import pandas as pd

# Load your data
data = pd.read_csv('your_molecules.csv')

# Generate fingerprints (hashed mode, default)
fingerprints, fp_generator = generate_fingerprints(
    data,
    smiles_column='smiles',
    radius=1,
    remove_zero_columns=True
)

# Save fingerprints to CSV
save_fingerprints(
    fingerprints,
    fp_generator,
    output_path='path/to/output.csv',
    include_header=True
)
```

### Unfolded Mode

Unfolded mode maps each unique substructure to its own column, producing the
smallest possible collision-free fingerprint. This is recommended for large
datasets or when interpretability is important.

```python
from bit_collision_free_MF import generate_fingerprints

# Generate fingerprints in unfolded mode
fingerprints, fp_generator = generate_fingerprints(
    data,
    smiles_column='smiles',
    radius=2,
    mode="unfolded",
    remove_zero_columns=True
)

# Interpretability: inspect what substructure each column represents
mapping = fp_generator.get_invariant_mapping()  # {col_index: invariant_id}

# Visualize a specific substructure with RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

inv_id = mapping[42]  # invariant ID for column 42
mol = Chem.MolFromSmiles("c1ccccc1O")
bi = {}
fp = AllChem.GetMorganFingerprint(mol, radius=2, bitInfo=bi)
if inv_id in bi:
    img = Draw.DrawMorganBit(mol, inv_id, bi)
```

### Using the CollisionFreeMorganFP Class Directly

```python
from bit_collision_free_MF import CollisionFreeMorganFP
import pandas as pd

# Load your data
data = pd.read_csv('your_molecules.csv')
smiles_list = data['smiles'].tolist()

# Create and fit the fingerprint generator
fp_generator = CollisionFreeMorganFP(radius=1)           # hashed mode (default)
# fp_generator = CollisionFreeMorganFP(radius=1, mode="unfolded")  # or unfolded mode
fp_generator.fit(smiles_list, remove_zero_columns=True)

# Generate fingerprints
fingerprints = fp_generator.transform(smiles_list)

# Get feature names (fp_0, fp_1, ... aligned with bit indices)
feature_names = fp_generator.get_feature_names()

# Create a DataFrame with the fingerprints
result_df = pd.DataFrame(fingerprints, columns=feature_names)
result_df.to_csv('fingerprints.csv', index=False)
```

### Train/Test Split with Consistent Dimensions

```python
from bit_collision_free_MF import CollisionFreeMorganFP

# fit() on training set records which columns are all-zero
fp_gen = CollisionFreeMorganFP(radius=1, mode="unfolded")
fp_gen.fit(train_smiles, remove_zero_columns=True)

# transform() reuses the same mapping and zero-column mask for both sets
X_train = fp_gen.transform(train_smiles)
X_test = fp_gen.transform(test_smiles)
# X_train.shape[1] == X_test.shape[1] guaranteed
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For academic inquiries or collaboration, please contact:
- Shifa Zhong (sfzhong@tongji.edu.cn)
- Jibai Li (51263903065@stu.ecnu.edu.cn)
