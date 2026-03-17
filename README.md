# bit_collision_free_MF

A Python package for generating molecular fingerprints without bit collisions.

## Description

`bit_collision_free_MF` generates **count-based** Morgan fingerprints while eliminating bit collisions, which can significantly improve the accuracy and reliability of molecular fingerprints in cheminformatics applications. The package automatically determines the optimal fingerprint length to ensure that each structural feature maps to a unique bit in the fingerprint.

Collision detection is based on comparing Morgan **invariants** (unique substructure identifiers), which catches all collision types — including collisions between different substructures at the same radius. This works reliably for all radius values, including radius=0.

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

- **Invariant-based collision detection**: compares actual substructure identifiers, not just radii, to guarantee truly collision-free fingerprints
- **Count-based output**: generates count fingerprints (not binary), preserving substructure frequency information
- Automatically determines the optimal fingerprint length using exponential growth + binary search
- Supports all radius values including radius=0
- **Consistent zero-column removal**: columns identified during `fit()` are reused in `transform()`, ensuring train/test dimensionality alignment
- Feature names (`fp_0`, `fp_1`, ...) are 0-indexed to match bit positions in RDKit's `bitInfo`, enabling correct substructure interpretation
- Easy CSV export with customizable headers
- Seamless integration with pandas and NumPy

## Usage

### Basic Usage

```python
from bit_collision_free_MF import generate_fingerprints, save_fingerprints
import pandas as pd

# Load your data
data = pd.read_csv('your_molecules.csv')

# Generate fingerprints
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

### Using the CollisionFreeMorganFP Class Directly

```python
from bit_collision_free_MF import CollisionFreeMorganFP
import pandas as pd

# Load your data
data = pd.read_csv('your_molecules.csv')
smiles_list = data['smiles'].tolist()

# Create and fit the fingerprint generator
fp_generator = CollisionFreeMorganFP(radius=1)
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
fp_gen = CollisionFreeMorganFP(radius=1)
fp_gen.fit(train_smiles, remove_zero_columns=True)

# transform() reuses the same zero-column mask for both sets
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
