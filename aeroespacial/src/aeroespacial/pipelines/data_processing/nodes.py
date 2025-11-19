import os
from typing import List

import pandas as pd


def read_and_prepare_files(
    directory: str, file_ids: List[int], base_columns: List[str]
) -> List[pd.DataFrame]:
    """
    Reads a series of files, renames their columns with a unique ID,
    and returns a list of the prepared DataFrames.

    Args:
        directory: The path to the directory containing the files.
        file_ids: A list of integers representing the file IDs to process.
        base_columns: A list of base column names to be used for renaming.

    Returns:
        A list of pandas DataFrames, each with uniquely named columns.
    """
    prepared_dfs = []

    for i in file_ids:
        file_name = f"cl_cmb_c501_m{i}.dat"
        file_path = os.path.join(directory, file_name)

        try:
            df = pd.read_csv(file_path, sep=r"\s+", header=None)

            # Create a unique list of column names for this file
            unique_columns = [f"{col}_{i}" for col in base_columns]
            df.columns = unique_columns

            prepared_dfs.append(df)
        except FileNotFoundError:
            print(f"Error: The file {file_name} was not found.")
        except Exception as e:
            print(f"An error occurred while reading {file_name}: {e}")

    return prepared_dfs


def combine_dataframes_horizontally(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combines a list of DataFrames side-by-side using pd.concat with axis=1.

    Args:
        dfs: A list of pandas DataFrames to be joined.

    Returns:
        A single DataFrame resulting from the horizontal concatenation.
    """
    # Join all DataFrames side-by-side using axis=1
    combined_df = pd.concat(dfs, axis=1)

    return combined_df


def calculate_average_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the row-wise average of 'cl_e' and 'cl_b' columns
    and returns a new DataFrame with the average columns.

    Args:
        df: The input DataFrame containing columns with 'cl_e_' and 'cl_b_' prefixes.

    Returns:
        A new DataFrame with two columns: 'cl_e_avg' and 'cl_b_avg'.
    """
    # Select all columns that start with 'cl_e_'
    cl_e_cols = [col for col in df.columns if col.startswith("cl_e_")]

    # Select all columns that start with 'cl_b_'
    cl_b_cols = [col for col in df.columns if col.startswith("cl_b_")]

    # Calculate the average for each group across rows (axis=1)
    cl_e_avg = df[cl_e_cols].mean(axis=1)
    cl_b_avg = df[cl_b_cols].mean(axis=1)

    # Create the new DataFrame with the two average columns
    final_df = pd.DataFrame({"cl_e_avg": cl_e_avg, "cl_b_avg": cl_b_avg})

    return final_df


def create_transposed_dataframe(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Transposes a single column from a DataFrame.

    Args:
        df: The input DataFrame.
        column_name: The name of the column to transpose.

    Returns:
        The transposed DataFrame.
    """
    transposed_df = df[[column_name]].T
    return transposed_df


def _is_true(x: pd.Series) -> pd.Series:
    return x == "t"


def _parse_percentage(x: pd.Series) -> pd.Series:
    x = x.str.replace("%", "")
    x = x.astype(float) / 100
    return x


def _parse_money(x: pd.Series) -> pd.Series:
    x = x.str.replace("$", "").str.replace(",", "")
    x = x.astype(float)
    return x


def preprocess_companies(companies: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for companies.

    Args:
        companies: Raw data.
    Returns:
        Preprocessed data, with `company_rating` converted to a float and
        `iata_approved` converted to boolean.
    """
    companies["iata_approved"] = _is_true(companies["iata_approved"])
    companies["company_rating"] = _parse_percentage(companies["company_rating"])
    return companies


def preprocess_shuttles(shuttles: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for shuttles.

    Args:
        shuttles: Raw data.
    Returns:
        Preprocessed data, with `price` converted to a float and `d_check_complete`,
        `moon_clearance_complete` converted to boolean.
    """
    shuttles["d_check_complete"] = _is_true(shuttles["d_check_complete"])
    shuttles["moon_clearance_complete"] = _is_true(shuttles["moon_clearance_complete"])
    shuttles["price"] = _parse_money(shuttles["price"])
    return shuttles


def create_model_input_table(
    shuttles: pd.DataFrame, companies: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        shuttles: Preprocessed data for shuttles.
        companies: Preprocessed data for companies.
        reviews: Raw data for reviews.
    Returns:
        Model input table.

    """
    rated_shuttles = shuttles.merge(reviews, left_on="id", right_on="shuttle_id")
    rated_shuttles = rated_shuttles.drop("id", axis=1)
    model_input_table = rated_shuttles.merge(
        companies, left_on="company_id", right_on="id"
    )
    model_input_table = model_input_table.dropna()
    return model_input_table
