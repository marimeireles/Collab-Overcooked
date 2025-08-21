import json
import os
import time
from argparse import ArgumentParser

import pandas as pd

level_1 = [
    "baked_bell_pepper",
    "baked_sweet_potato",
    "boiled_egg",
    "boiled_mushroom",
    "boiled_sweet_potato",
]

level_2 = [
    "baked_potato_slices",
    "baked_pumpkin_slices",
    "boiled_corn_slices",
    "boiled_green_bean_slices",
    "boiled_potato_slices",
]

level_3 = [
    "baked_bell_pepper_soup",
    "baked_carrot_soup",
    "baked_mushroom_soup",
    "baked_potato_soup",
    "baked_pumpkin_soup",
]

level_4 = [
    "sliced_bell_pepper_and_corn_stew",
    "sliced_bell_pepper_and_lentil_stew",
    "sliced_eggplant_and_chickpea_stew",
    "sliced_pumpkin_and_chickpea_stew",
    "sliced_zucchini_and_chickpea_stew",
]

level_5 = [
    "mashed_broccoli_and_bean_patty",
    "mashed_carrot_and_chickpea_patty",
    "mashed_cauliflower_and_lentil_patty",
    "mashed_potato_and_pea_patty",
    "mashed_sweet_potato_and_bean_patty",
]

level_6 = [
    "potato_carrot_and_onion_patty",
    "romaine_lettuce_pea_and_tomato_patty",
    "sweet_potato_spinach_and_mushroom_patty",
    "taro_bean_and_bell_pepper_patty",
    "zucchini_green_pea_and_onion_patty",
]

levels = {
    "level_1": level_1,
    "level_2": level_2,
    "level_3": level_3,
    "level_4": level_4,
    "level_5": level_5,
    "level_6": level_6,
}


def extract_experiment_name(custom_dir):
    """Extract experiment name from directory path for consistent filename generation."""
    if not custom_dir:
        return "default"
        
    path_parts = custom_dir.strip('/').split('/')
    
    # Find the experiment directory name (starts with 'experiment')
    experiment_name = None
    for part in path_parts:
        if part.startswith('experiment'):
            experiment_name = part
            break
    
    # If no experiment name found, use the last directory name
    if experiment_name is None:
        experiment_name = path_parts[-1] if path_parts else "default"
    
    return experiment_name


def get_level(order):
    for level_name, orders in levels.items():
        if order in orders:
            return level_name
    return None


def main(custom_dir=None):
    # Extract experiment name for filename
    experiment_name = extract_experiment_name(custom_dir)
    
    # Always use experiment-specific output filename
    output_file = f"eval_result/converted_data_{experiment_name}.csv"
    
    # Try experiment-specific input file first
    input_file = f"eval_result/statistics_data_{experiment_name}.csv"
    
    # For backward compatibility, try the default filename if experiment-specific file doesn't exist
    if not os.path.exists(input_file):
        input_file = "eval_result/statistics_data.csv"
        print(f"⚠️  Experiment-specific file not found, trying default: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file {input_file} does not exist!")
        return
    
    print(f"📊 Reading data from: {input_file}")
    data = pd.read_csv(input_file)

    data["level"] = data["order"].apply(get_level)

    # Warn and preserve any orders not mapped to a level
    unmapped_mask = data["level"].isna()
    if unmapped_mask.any():
        missing_orders = sorted(data.loc[unmapped_mask, "order"].unique().tolist())
        print(
            f"⚠️  {len(missing_orders)} orders are not mapped to a level and will be grouped under 'unmapped': {missing_orders}"
        )
        data.loc[unmapped_mask, "level"] = "unmapped"

    columns_to_average = [
        col for col in data.columns if col not in ["model", "order", "level"]
    ]

    data[columns_to_average] = data[columns_to_average].apply(
        pd.to_numeric, errors="coerce"
    )

    grouped_data = (
        data.groupby(["model", "level"], dropna=False)[columns_to_average]
            .mean()
            .reset_index()
    )

    grouped_data.to_csv(output_file, sep=",", index=False)
    print(f"✅ Converted data saved to: {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert statistics data by levels")
    parser.add_argument(
        "--custom_dir",
        type=str,
        default=None,
        help="Custom directory path to extract experiment name from",
    )
    
    args = parser.parse_args()
    main(args.custom_dir)
