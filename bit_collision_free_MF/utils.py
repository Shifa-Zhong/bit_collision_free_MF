"""
Utility functions for bit_collision_free_MF package.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, List, Tuple, Any


def _has_collision(smiles_list: List[str], radius: int, length: int) -> bool:
    """
    Check whether any bit collision exists at the given fingerprint length.

    Parameters
    ----------
    smiles_list : List[str]
        A list of SMILES strings to check.
    radius : int
        The radius for the Morgan fingerprint algorithm.
    length : int
        The fingerprint bit vector length to test.

    Returns
    -------
    bool
        True if at least one collision is detected.
    """
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue

        bit_info: Dict[int, List[Tuple[int, int]]] = {}
        _ = AllChem.GetMorganFingerprintAsBitVect(mol, radius, length, bitInfo=bit_info)

        for k in bit_info:
            path_info = [bit_info[k][i][1] for i in range(len(bit_info[k]))]
            if len(set(path_info)) > 1:
                return True
    return False


def get_optimized_length(smiles_list: List[str], radius: int = 1) -> int:
    """
    Find the optimal fingerprint length to avoid bit collisions.

    Uses a two-phase strategy: first grows exponentially (doubling) to find an
    upper bound with no collisions, then binary-searches between the last
    colliding length and the first collision-free length to find the smallest
    collision-free value.

    Parameters
    ----------
    smiles_list : List[str]
        A list of SMILES strings to analyze.
    radius : int, default=1
        The radius for the Morgan fingerprint algorithm.

    Returns
    -------
    int
        The optimal length for collision-free fingerprints.
    """
    # Phase 1: exponential growth to find a collision-free upper bound
    n = 100
    while _has_collision(smiles_list, radius, n):
        n *= 2

    # Phase 2: binary search between n//2 (last known collision) and n
    lo = max(100, n // 2)
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_collision(smiles_list, radius, mid):
            lo = mid + 1
        else:
            hi = mid

    return lo


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
