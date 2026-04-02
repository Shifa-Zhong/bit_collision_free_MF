"""
Fingerprint generation module.

This module provides the core functionality for generating
collision-free Morgan fingerprints for molecules.
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Tuple, Dict, Optional, Union, Any

from .utils import get_optimized_length


class CollisionFreeMorganFP:
    """
    A class to generate collision-free Morgan fingerprints.

    This class implements a method to create Morgan fingerprints without bit collisions.
    Two modes are supported:

    - ``"hashed"`` (default): Uses RDKit's hashed Morgan fingerprint and automatically
      finds the minimal fingerprint length that avoids all bit collisions.
    - ``"unfolded"``: Builds a direct mapping from Morgan invariant IDs to column
      indices. Each unique substructure gets its own column, guaranteeing zero
      collisions with the minimal possible number of columns.

    Parameters
    ----------
    radius : int, default=1
        The radius for the Morgan fingerprint algorithm.
    length : Optional[int], default=None
        The length of the fingerprint bit vector. If None, it will be automatically
        determined to avoid bit collisions. Ignored in ``"unfolded"`` mode.
    mode : str, default="hashed"
        ``"hashed"`` for optimized-length hashed fingerprints,
        ``"unfolded"`` for direct invariant-to-column mapping.
    """

    def __init__(self, radius: int = 1, length: Optional[int] = None,
                 mode: str = "hashed"):
        """Initialize the CollisionFreeMorganFP class."""
        if mode not in ("hashed", "unfolded"):
            raise ValueError(f"mode must be 'hashed' or 'unfolded', got '{mode}'")
        self.radius = radius
        self.length = length
        self.mode = mode
        self._zero_columns: List[int] = []
        self._is_fitted: bool = False
        self._invariant_to_col: Dict[int, int] = {}  # unfolded mode only

    def fit(self, smiles_list: List[str],
            remove_zero_columns: bool = False) -> 'CollisionFreeMorganFP':
        """
        Determine the optimal fingerprint length to avoid bit collisions
        and optionally record zero columns for consistent removal.

        Parameters
        ----------
        smiles_list : List[str]
            A list of SMILES strings to analyze.
        remove_zero_columns : bool, default=False
            Whether to identify and record zero columns for later removal
            in transform().

        Returns
        -------
        CollisionFreeMorganFP
            The fitted object.
        """
        if self.mode == "unfolded":
            all_invariants: set = set()
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                unhashed_fp = AllChem.GetMorganFingerprint(mol, self.radius)
                all_invariants.update(unhashed_fp.GetNonzeroElements().keys())
            sorted_invariants = sorted(all_invariants)
            self._invariant_to_col = {inv: idx for idx, inv in enumerate(sorted_invariants)}
            self.length = len(sorted_invariants)
        else:
            if self.length is None:
                self.length = get_optimized_length(smiles_list, self.radius)

        if remove_zero_columns:
            fingerprints = np.vstack([self._get_fingerprint(s) for s in smiles_list])
            sum_cols = fingerprints.sum(axis=0)
            self._zero_columns = np.where(sum_cols == 0)[0].tolist()
        else:
            self._zero_columns = []

        self._is_fitted = True
        return self

    def _get_fingerprint(self, smiles: str) -> np.ndarray:
        """
        Generate a Morgan fingerprint for a single molecule.

        Parameters
        ----------
        smiles : str
            SMILES string of the molecule.

        Returns
        -------
        np.ndarray
            The fingerprint as a numpy array.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES string: {smiles}")

        if self.mode == "unfolded":
            fp_array = np.zeros(self.length, dtype=np.int32)
            unhashed_fp = AllChem.GetMorganFingerprint(mol, self.radius)
            for inv, count in unhashed_fp.GetNonzeroElements().items():
                if inv in self._invariant_to_col:
                    fp_array[self._invariant_to_col[inv]] = count
            return fp_array
        else:
            fp = AllChem.GetHashedMorganFingerprint(mol, self.radius, self.length)
            return np.array(list(fp))

    def transform(self,
                  smiles_list: List[str],
                  remove_zero_columns: bool = False) -> np.ndarray:
        """
        Generate fingerprints for a list of SMILES strings.

        If zero columns were recorded during fit(), they are always removed
        from the output to ensure consistent dimensionality between training
        and test sets. The ``remove_zero_columns`` parameter here additionally
        removes columns that are all zeros in the *current* batch (only when
        fit() did not already record zero columns).

        Parameters
        ----------
        smiles_list : List[str]
            A list of SMILES strings to convert to fingerprints.
        remove_zero_columns : bool, default=False
            Whether to remove columns that are all zeros in this batch.
            Ignored if zero columns were already recorded during fit().

        Returns
        -------
        np.ndarray
            A 2D array of fingerprints, where each row is a fingerprint.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform().")

        fingerprints = np.vstack([self._get_fingerprint(s) for s in smiles_list])

        # Use zero columns from fit() if available, otherwise optionally detect from this batch
        zero_cols = self._zero_columns
        if not zero_cols and remove_zero_columns:
            sum_cols = fingerprints.sum(axis=0)
            zero_cols = np.where(sum_cols == 0)[0].tolist()

        if zero_cols:
            mask = np.ones(fingerprints.shape[1], dtype=bool)
            mask[zero_cols] = False
            fingerprints = fingerprints[:, mask]

        return fingerprints

    def fit_transform(self,
                      smiles_list: List[str],
                      remove_zero_columns: bool = False) -> np.ndarray:
        """
        Fit the model and generate fingerprints.

        Parameters
        ----------
        smiles_list : List[str]
            A list of SMILES strings to analyze and convert.
        remove_zero_columns : bool, default=False
            Whether to remove columns that are all zeros.

        Returns
        -------
        np.ndarray
            A 2D array of fingerprints, where each row is a fingerprint.
        """
        self.fit(smiles_list, remove_zero_columns=remove_zero_columns)
        return self.transform(smiles_list, remove_zero_columns)

    def get_feature_names(self) -> List[str]:
        """
        Get the feature names for the fingerprint columns.

        Returns
        -------
        List[str]
            A list of feature names in the format fp_1, fp_2, etc.
        """
        if not self._is_fitted or self.length is None:
            raise ValueError("Model must be fitted before getting feature names")

        all_names = [f'fp_{i}' for i in range(self.length)]

        if self._zero_columns:
            return [name for i, name in enumerate(all_names) if i not in self._zero_columns]

        return all_names

    def get_invariant_mapping(self) -> Dict[int, int]:
        """
        Get the mapping from column index to Morgan invariant ID.

        Only available in ``"unfolded"`` mode. This mapping enables
        interpretability: use the invariant ID with RDKit's ``bitInfo``
        to identify the exact substructure each column represents.

        Returns
        -------
        Dict[int, int]
            A dictionary mapping column index to Morgan invariant ID.

        Raises
        ------
        ValueError
            If called in ``"hashed"`` mode.
        RuntimeError
            If the model has not been fitted yet.

        Examples
        --------
        >>> from rdkit.Chem import Draw, AllChem, Chem
        >>> mapping = fp_gen.get_invariant_mapping()
        >>> # Suppose column 42 is important; find its invariant ID
        >>> inv_id = mapping[42]
        >>> # Visualize the substructure on a molecule
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> bi = {}
        >>> fp = AllChem.GetMorganFingerprint(mol, radius=2, bitInfo=bi)
        >>> if inv_id in bi:
        ...     img = Draw.DrawMorganBit(mol, inv_id, bi)
        """
        if self.mode != "unfolded":
            raise ValueError("get_invariant_mapping() is only available in 'unfolded' mode")
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before get_invariant_mapping()")
        return {idx: inv for inv, idx in self._invariant_to_col.items()}


def generate_fingerprints(
        data: Union[pd.DataFrame, List[str]],
        smiles_column: Optional[str] = None,
        radius: int = 1,
        length: Optional[int] = None,
        remove_zero_columns: bool = False,
        mode: str = "hashed"
) -> Tuple[np.ndarray, CollisionFreeMorganFP]:
    """
    Generate collision-free Morgan fingerprints from molecular data.

    Parameters
    ----------
    data : Union[pd.DataFrame, List[str]]
        Either a DataFrame containing SMILES strings or a list of SMILES strings.
    smiles_column : Optional[str], default=None
        The name of the column containing SMILES strings if data is a DataFrame.
        Required if data is a DataFrame.
    radius : int, default=1
        The radius for the Morgan fingerprint algorithm.
    length : Optional[int], default=None
        The length of the fingerprint bit vector. If None, it will be automatically
        determined to avoid bit collisions. Ignored in ``"unfolded"`` mode.
    remove_zero_columns : bool, default=False
        Whether to remove columns that are all zeros.
    mode : str, default="hashed"
        ``"hashed"`` for optimized-length hashed fingerprints,
        ``"unfolded"`` for direct invariant-to-column mapping.

    Returns
    -------
    Tuple[np.ndarray, CollisionFreeMorganFP]
        A tuple containing:
        - A 2D array of fingerprints, where each row is a fingerprint.
        - The fitted CollisionFreeMorganFP object.

    Raises
    ------
    ValueError
        If data is a DataFrame but smiles_column is not provided.
    """
    if isinstance(data, pd.DataFrame):
        if smiles_column is None:
            raise ValueError("smiles_column must be provided when data is a DataFrame")
        smiles_list = data[smiles_column].tolist()
    else:
        smiles_list = data

    fp_generator = CollisionFreeMorganFP(radius=radius, length=length, mode=mode)
    fingerprints = fp_generator.fit_transform(smiles_list, remove_zero_columns)

    return fingerprints, fp_generator


def save_fingerprints(
        fingerprints: np.ndarray,
        fp_generator: CollisionFreeMorganFP,
        output_path: str = "fingerprints.csv",
        include_header: bool = True,
        index: bool = False
) -> None:
    """
    Save generated fingerprints to a CSV file.

    Parameters
    ----------
    fingerprints : np.ndarray
        The 2D array of fingerprints to save.
    fp_generator : CollisionFreeMorganFP
        The fitted fingerprint generator object.
    output_path : str, default="fingerprints.csv"
        The path where the CSV file will be saved.
    include_header : bool, default=True
        Whether to include a header with column names in the format fp_1, fp_2, etc.
    index : bool, default=False
        Whether to include an index column in the output CSV.

    Returns
    -------
    None
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    columns = None
    if include_header:
        columns = fp_generator.get_feature_names()

    df = pd.DataFrame(fingerprints, columns=columns)
    df.to_csv(output_path, index=index)
