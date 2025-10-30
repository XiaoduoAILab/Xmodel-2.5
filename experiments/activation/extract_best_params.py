#!/usr/bin/env python3
import sys

import pandas as pd


def extract_best_parameters(csv_file):
    """Extract best parameters from HPO results CSV file"""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Find the row with the minimum value (best result)
        best_row = df.loc[df['value'].idxmin()]

        print("===== Optimization Completed =====")
        print(f"Best trial: {int(best_row['number'])}")
        print(f"Best RMS distance: {best_row['value']:.4f}")
        print("Best parameters:")

        # Extract parameter columns (those starting with 'params_')
        param_columns = [col for col in df.columns if col.startswith('params_')]

        for param_col in param_columns:
            param_name = param_col.replace('params_', '')
            param_value = best_row[param_col]
            print(f"  {param_name}: {param_value}")

        print(f"\nResults loaded from {csv_file}")

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_best_params.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    extract_best_parameters(csv_file)
