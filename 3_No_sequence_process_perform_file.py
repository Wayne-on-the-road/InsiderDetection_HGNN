import os
import pandas as pd

def parse_filename(file_name):
    """
    Parse the filename to extract different settings and their values.
    '-' separates different settings and '_' separates the setting name from its value.
    """
    # Remove the 'total_result-' prefix and '.csv' suffix
    base_name = file_name.replace('total_result-', '').replace('.csv', '')

    # Split by '-'
    settings = base_name.split('-')

    # Parse each setting into a key-value pair
    settings_dict = {}
    for setting in settings:
        key, value = setting.split('_', 1)  # Split by the first occurrence of '_'
        settings_dict[key] = value

    return settings_dict

def process_file(file_path):
    """
    Process a single CSV file, calculate mean, std, min, max, median for performance metrics,
    grouped by 'relation' and 'GNN_model', and return a DataFrame with the training round number
    for max, min, and median values.
    """
    # Extract settings from file name
    file_name = os.path.basename(file_path)
    settings = parse_filename(file_name)

    # Read the CSV content
    df = pd.read_csv(file_path)

    # Check if the DataFrame is empty
    if df.empty:
        print(f"Warning: {file_path} is empty.")
        return None

    # Check if 'relation' and 'GNN_model' columns exist before accessing them
    if 'relation' not in df.columns or 'GNN_model' not in df.columns:
        print(f"Warning: {file_path} is missing 'relation' or 'GNN_model' columns.")
        return None

    # Group by 'relation' and 'GNN_model'
    grouped = df.groupby(['relation', 'GNN_model'])

    metrics = ['Acc', 'Pre', 'Rec', 'F1', 'Auc']
    results = []

    for (relation, gnn_model), group in grouped:
        metrics_mean = {}
        metrics_std = {}
        metrics_min = {}
        metrics_max = {}
        metrics_median = {}
        metrics_min_round = {}
        metrics_max_round = {}
        metrics_median_round = {}

        for metric in metrics:
            non_zero_values = group[group[metric] != 0]  # Filter out zero values
            if len(non_zero_values) > 0:  # Only calculate if there are non-zero values
                metrics_mean[metric] = non_zero_values[metric].mean()
                metrics_std[metric] = non_zero_values[metric].std()
                metrics_min[metric] = non_zero_values[metric].min()
                metrics_max[metric] = non_zero_values[metric].max()
                metrics_median[metric] = non_zero_values[metric].median()

                # Get the 'round' number where max, min, and median occur
                metrics_min_round[metric] = non_zero_values.loc[non_zero_values[metric].idxmin(), 'round']
                metrics_max_round[metric] = non_zero_values.loc[non_zero_values[metric].idxmax(), 'round']
                metrics_median_round[metric] = non_zero_values.loc[(non_zero_values[metric] - metrics_median[metric]).abs().idxmin(), 'round']
            else:
                # No valid data, set all values to None
                metrics_mean[metric] = None
                metrics_std[metric] = None
                metrics_min[metric] = None
                metrics_max[metric] = None
                metrics_median[metric] = None
                metrics_min_round[metric] = None
                metrics_max_round[metric] = None
                metrics_median_round[metric] = None

        # Create a row for the result based on 'relation' and 'GNN_model'
        result = {**settings}
        result['relation'] = relation
        result['GNN_model'] = gnn_model

        # Add mean, std, min, max, median, and the 'round' numbers for each performance metric
        for metric in metrics:
            result[f'{metric}_mean'] = metrics_mean[metric]
            result[f'{metric}_std'] = metrics_std[metric]
            result[f'{metric}_min'] = metrics_min[metric]
            result[f'{metric}_min_round'] = metrics_min_round[metric]
            result[f'{metric}_max'] = metrics_max[metric]
            result[f'{metric}_max_round'] = metrics_max_round[metric]
            result[f'{metric}_median'] = metrics_median[metric]
            result[f'{metric}_median_round'] = metrics_median_round[metric]

        results.append(result)

    return pd.DataFrame(results)

def process_all_files(root_dir, output_file):
    """
    Find all CSV files in subdirectories, process each one, and save the results in a new CSV file.
    """
    all_results = []

    # Traverse the root directory to find all CSV files
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv') and 'total_result' in file:
                file_path = os.path.join(subdir, file)
                file_results = process_file(file_path)
                if file_results is not None:
                    all_results.append(file_results)

    if all_results:
        # Concatenate all results into a single DataFrame
        final_df = pd.concat(all_results, ignore_index=True)

        # Define main and metric columns for organization
        main_columns = ['relation', 'GNN_model']
        metric_columns = []
        for metric in ['Acc', 'Pre', 'Rec', 'F1', 'Auc']:
            metric_columns.extend([
                f'{metric}_mean', f'{metric}_std', f'{metric}_min', f'{metric}_min_round',
                f'{metric}_max', f'{metric}_max_round', f'{metric}_median', f'{metric}_median_round'
            ])
        other_columns = [col for col in final_df.columns if col not in main_columns + metric_columns]

        # Reorder columns for final output
        final_df = final_df[main_columns + other_columns + metric_columns]

        # Append directory name to output file name
        lowest_level_dir = os.path.basename(os.path.normpath(root_dir))
        output_file = output_file.replace('.csv', f'_{lowest_level_dir}.csv')

        # Save to CSV
        final_df.to_csv(output_file, index=False)
        print(f'Results saved to {output_file}')
    else:
        print("No valid results found.")

# Example usage
root_directory = r'./result_root_file'
output_csv = os.path.join(root_directory, 'aggregated_results.csv')
process_all_files(root_directory, output_csv)
