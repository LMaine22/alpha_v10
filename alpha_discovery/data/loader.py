# alpha_discovery/data/loader.py

import os
import pandas as pd
from ..config import settings  # Note the relative import from the parent package

# The 5th row in Excel is index 4 for programming (0-indexed)
HEADER_ROW = 4


def convert_excel_to_parquet():
    """
    Loads data from the source Excel file, combines all sheets,
    and saves the result to a more efficient Parquet file.

    This is a one-time operation.
    """
    excel_path = settings.data.excel_file_path
    parquet_path = settings.data.parquet_file_path

    if not os.path.exists(excel_path):
        print(f" Error: Excel file not found at '{excel_path}'")
        return

    print(f" Loading Excel file from '{excel_path}'...")
    try:
        xls = pd.ExcelFile(excel_path)

        all_dfs = []
        for sheet_name in xls.sheet_names:
            # Read the sheet, assuming the header is at the specified row
            df = pd.read_excel(xls, sheet_name=sheet_name, header=HEADER_ROW)

            # Standardize the date column name
            if 'Dates' in df.columns:
                df = df.rename(columns={'Dates': 'Date'})

            if 'Date' not in df.columns:
                print(f" Warning: Sheet '{sheet_name}' is missing the 'Date' column. Skipping.")
                continue

            # Drop rows where the Date is missing
            df.dropna(subset=['Date'], inplace=True)

            # Make 'Date' the index
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')

            # Prepend the sheet name (ticker) to each column
            df = df.add_prefix(f"{sheet_name}_")
            all_dfs.append(df)

        print("Combining data from all sheets...")
        # Combine all dataframes, aligning by the 'Date' index
        combined_df = pd.concat(all_dfs, axis=1, join='outer')

        # Sort by date and forward-fill missing values
        combined_df = combined_df.sort_index()
        combined_df.ffill(inplace=True)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)

        print(f" Saving combined data to Parquet file at '{parquet_path}'...")
        combined_df.to_parquet(parquet_path)
        print(f" Conversion successful! Shape of saved data: {combined_df.shape}")

    except Exception as e:
        print(f" An error occurred during conversion: {e}")


def load_data_from_parquet() -> pd.DataFrame:
    """
    Loads the processed data from the Parquet file.
    This should be used for all subsequent data loading in the project.
    """
    parquet_path = settings.data.parquet_file_path

    if not os.path.exists(parquet_path):
        print(f" Parquet file not found at '{parquet_path}'.")
        print("Please run the conversion from Excel first.")
        return pd.DataFrame()

    print(f"Loading data from '{parquet_path}'...")
    df = pd.read_parquet(parquet_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


if __name__ == '__main__':
    # This allows you to run the script directly from your terminal
    # to perform the one-time data conversion.
    # Example command: python -m alpha_discovery.data.loader
    convert_excel_to_parquet()