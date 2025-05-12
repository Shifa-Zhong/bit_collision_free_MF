# bit_collision_free_MF

A Python package for generating molecular fingerprints without bit collisions.

## Description

`bit_collision_free_MF` generates Morgan fingerprints while eliminating bit collisions, which can significantly improve the accuracy and reliability of molecular fingerprints in cheminformatics applications. The package automatically determines the optimal fingerprint length to ensure that each structural feature maps to a unique bit in the fingerprint.

## Installation

### Requirements

- Python 3.9 or higher
- numpy
- pandas
- rdkit

### Simple Installation

```bash
pip install bit_collision_free_MF
```

This will automatically install all dependencies, including RDKit.

### Manual Installation

```bash
# Install dependencies
pip install numpy pandas rdkit

# Install the package
pip install bit_collision_free_MF
```

For development installation:
```bash
# Clone the repository
git clone https://github.com/yourusername/bit_collision_free_MF.git
cd bit_collision_free_MF

# Install in development mode
pip install -e .
```

### Troubleshooting

If you encounter issues installing RDKit:

1. **Verify Python version**: This package requires Python 3.9 or higher.
2. **Alternative installation methods**:
   - For older Python versions: `pip install rdkit-pypi`
   - Using conda: `conda install -c conda-forge rdkit`

## Features

- Automatically determines the optimal fingerprint length to avoid bit collisions
- Supports custom fingerprint radius
- Option to remove zero-value columns
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
fp_generator.fit(smiles_list)

# Generate fingerprints
fingerprints = fp_generator.transform(smiles_list, remove_zero_columns=True)

# Get feature names
feature_names = fp_generator.get_feature_names()

# Create a DataFrame with the fingerprints
result_df = pd.DataFrame(fingerprints, columns=feature_names)

# Save to CSV
result_df.to_csv('fingerprints.csv', index=False)
```

## License

This software is currently not open source. All rights reserved. Redistribution, modification, or use of this software in any form is not permitted until the associated research article is formally accepted and published.

Upon acceptance, the software will be released under the MIT License.

## Contact

For academic inquiries or collaboration, please contact:
- Shifa Zhong (sfzhong@tongji.edu.cn)
- Jibai Li (51263903065@stu.ecnu.edu.cn)