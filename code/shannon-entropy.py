import math
from collections import Counter
import pandas as pd


def extract_data():
    """
    Load the data and clean it
    :return: Dataframe of columns
    """
    df = pd.read_excel("../data/analyze.xlsx")
    columns = df.columns
    return df, columns


def shannon_entropy(data, unit):
    """
    Shannon Entropy Function
    :param data: Column data of a comparison factor
    :param unit: base of the logarithm (unit of measurement)
    :return: Returns the total shannon entropy of each comparison factor
    """
    # If the input is empty, return 0.0 because there's
    # no inconsistency - nothing to measure
    if not data:
        return 0.0
    # Count how often each number appears
    counts = Counter(data)
    # Calculate the total number of items in the data
    total = len(data)
    # Loop over each unique number's count
    # Compute its probability
    # Apply the shannon entropy formula for that number
    # Accumulate the total entropy for the comparison factor
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p, unit)
    return entropy


def shannon_entropy_results(df, columns):
    """
    Calculate the Shannon Entropy Results for each comparison factor
    :param df: Data frame of the dataset
    :param columns: Data for each column in the dataset
    :return: A dictionary with the Shannon Entropy Results for each comparison factor
    """
    results = {}
    for i in range(5, len(columns)):
        # Get the column data for each comparison factor
        column = columns[i]
        column_data = df[column]
        # Get the base of the logarithm that determines the unit of measurement for each factor
        log_base = int(column.split(" ")[1])
        # Convert all column data to strings
        string_data = column_data.astype(str).tolist()
        # Compute the shannon entropy value for a given factor
        shannon_entropy_value = round(shannon_entropy(string_data, log_base), 3)
        # Store result in the dictionary
        results[column] = shannon_entropy_value
    # Sort the dictionary by entropy value in ascending order
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]))
    return sorted_results


def export_shannon_entropy_results(results):
    entropy_df = pd.DataFrame({
        'Factor': results.keys(),
        'Entropy value': results.values()
    })
    output_file = "../output/entropy_results.xlsx"
    entropy_df.to_excel(output_file, index=False)
    print(f"Entropy results exported to '{output_file}'")


def main():
    dataframe, dataframe_columns = extract_data()
    entropy_data = shannon_entropy_results(dataframe, dataframe_columns)
    export_shannon_entropy_results(entropy_data)


if __name__ == "__main__":
    main()
