"""
Utility functions for bit_collision_free_MF package.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, List, Tuple, Any


def get_optimized_length(data: pd.DataFrame, radius: int = 1) -> int:
    """
    Find the optimal fingerprint length to avoid bit collisions.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing a 'smiles' column with SMILES strings.
    radius : int, default=1
        The radius for the Morgan fingerprint algorithm.

    Returns
    -------
    int
        The optimal length for collision-free fingerprints.
    """
    n = 100  # Starting length
    flag = True

    while flag:
        for smile in data['smiles']:
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                continue

            bit_info: Dict[int, List[Tuple[int, int]]] = {}
            # Get fingerprint with bit info
            _ = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n, bitInfo=bit_info)

            # Check for collisions
            flag = False
            for k in bit_info.keys():
                # Get the list of paths/radii for this bit
                path_info = [bit_info[k][i][1] for i in range(len(bit_info[k]))]
                # If there are different radii for the same bit, we have a collision
                if len(set(path_info)) > 1:
                    flag = True
                    break

            if flag:
                n += 10  # Increase fingerprint length
                break

    return n


def check_for_zero_columns(fingerprints: np.ndarray) -> List[int]:
    """
    Identify columns that are all zeros in the fingerprint matrix.

    Parameters
    ----------
    fingerprints : np.ndarray
        A 2D array of fingerprints.

    Returns
    -------
    List[int]
        The indices of columns that are all zeros.
    """
    sum_cols = fingerprints.sum(axis=0)
    return np.where(sum_cols == 0)[0].tolist()