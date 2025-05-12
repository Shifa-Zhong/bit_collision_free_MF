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

    This class implements a method to create Morgan fingerprints without bit collisions
    by determining the optimal bit vector length.

    Parameters
    ----------
    radius : int, default=1
        The radius for the Morgan fingerprint algorithm.
    length : Optional[int], default=None
        The length of the fingerprint bit vector. If None, it will be automatically
        determined to avoid bit collisions.
    """

    def __init__(self, radius: int = 1, length: Optional[int] = None):
        """Initialize the CollisionFreeMorganFP class."""
        self.radius = radius
        self.length = length
        self._zero_columns: List[int] = []

    def fit(self, smiles_list: List[str]) -> 'CollisionFreeMorganFP':
        """
        Determine the optimal fingerprint length to avoid bit collisions.

        Parameters
        ----------
        smiles_list : List[str]
            A list of SMILES strings to analyze.

        Returns
        -------
        CollisionFreeMorganFP
            The fitted object.
        """
        # Create a temporary dataframe to use with our utility function
        temp_df = pd.DataFrame({'smiles': smiles_list})

        # Special handling: if radius==0 and length is None, use radius=1 to get optimized length
        if self.length is None:
            if self.radius == 0:
                self.length = get_optimized_length(temp_df, 1)
            else:
                self.length = get_optimized_length(temp_df, self.radius)

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

        fp = AllChem.GetHashedMorganFingerprint(mol, self.radius, self.length)
        return np.array(list(fp))

    def transform(self,
                  smiles_list: List[str],
                  remove_zero_columns: bool = False) -> np.ndarray:
        """
        Generate fingerprints for a list of SMILES strings.

        Parameters
        ----------
        smiles_list : List[str]
            A list of SMILES strings to convert to fingerprints.
        remove_zero_columns : bool, default=False
            Whether to remove columns that are all zeros.

        Returns
        -------
        np.ndarray
            A 2D array of fingerprints, where each row is a fingerprint.
        """
        # Generate fingerprints for all molecules
        fingerprints = np.vstack([self._get_fingerprint(s) for s in smiles_list])

        # Identify and remove zero columns if requested
        if remove_zero_columns:
            sum_cols = fingerprints.sum(axis=0)
            self._zero_columns = np.where(sum_cols == 0)[0].tolist()

            # Create a mask to keep non-zero columns
            if self._zero_columns:
                mask = np.ones(fingerprints.shape[1], dtype=bool)
                mask[self._zero_columns] = False
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
        self.fit(smiles_list)
        return self.transform(smiles_list, remove_zero_columns)

    def get_feature_names(self) -> List[str]:
        """
        Get the feature names for the fingerprint columns.

        Returns
        -------
        List[str]
            A list of feature names in the format fp_1, fp_2, etc.
        """
        if not hasattr(self, 'length') or self.length is None:
            raise ValueError("Model must be fitted before getting feature names")

        # Generate names for all potential columns
        all_names = [f'fp_{i + 1}' for i in range(self.length)]

        # Remove names for zero columns if they were identified
        if self._zero_columns:
            return [name for i, name in enumerate(all_names) if i not in self._zero_columns]

        return all_names


def generate_fingerprints(
        data: Union[pd.DataFrame, List[str]],
        smiles_column: Optional[str] = None,
        radius: int = 1,
        length: Optional[int] = None,
        remove_zero_columns: bool = False
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
        determined to avoid bit collisions.
    remove_zero_columns : bool, default=False
        Whether to remove columns that are all zeros.

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
    # Extract SMILES list from input data
    if isinstance(data, pd.DataFrame):
        if smiles_column is None:
            raise ValueError("smiles_column must be provided when data is a DataFrame")
        smiles_list = data[smiles_column].tolist()
    else:
        smiles_list = data

    # Initialize and fit the fingerprint generator
    fp_generator = CollisionFreeMorganFP(radius=radius, length=length)
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
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare column names if headers are requested
    columns = None
    if include_header:
        columns = fp_generator.get_feature_names()

    # Convert fingerprints to DataFrame and save
    df = pd.DataFrame(fingerprints, columns=columns)
    df.to_csv(output_path, index=index)