"""
Utility functions for bit_collision_free_MF package.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Dict, List, Tuple, Set, Any


def _has_collision(smiles_list: List[str], radius: int, length: int) -> bool:
    """
    Check whether any bit collision exists at the given fingerprint length.

    Uses the unhashed Morgan fingerprint to obtain the true environment
    identifiers (invariants), then checks whether any two *distinct*
    invariants would be mapped to the same bit position (invariant % length).
    This approach detects all collision types, including same-radius
    collisions and works correctly even when radius=0.

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
    # Collect all unique invariants across the entire dataset
    all_invariants: Set[int] = set()
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue
        unhashed_fp = AllChem.GetMorganFingerprint(mol, radius)
        all_invariants.update(unhashed_fp.GetNonzeroElements().keys())

    # Check if any two distinct invariants map to the same bit
    bit_to_invariant: Dict[int, int] = {}
    for inv in all_invariants:
        bit = inv % length
        if bit in bit_to_invariant:
            if bit_to_invariant[bit] != inv:
                return True
        else:
            bit_to_invariant[bit] = inv

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
