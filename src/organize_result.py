import json
import os
import time
from argparse import ArgumentParser

import pandas as pd

models = ["gpt-4o"]

orders = [
    "baked_bell_pepper",
    "baked_sweet_potato",
    "boiled_egg",
    "boiled_mushroom",
    "boiled_sweet_potato",
    "baked_potato_slices",
    "baked_pumpkin_slices",
    "boiled_corn_slices",
    "boiled_green_bean_slices",
    "boiled_potato_slices",
    "baked_bell_pepper_soup",
    "baked_carrot_soup",
    "baked_mushroom_soup",
    "baked_potato_soup",
    "baked_pumpkin_soup",
    "sliced_bell_pepper_and_corn_stew",
    "sliced_bell_pepper_and_lentil_stew",
    "sliced_eggplant_and_chickpea_stew",
    "sliced_pumpkin_and_chickpea_stew",
    "sliced_zucchini_and_chickpea_stew",
    "mashed_broccoli_and_bean_patty",
    "mashed_carrot_and_chickpea_patty",
    "mashed_cauliflower_and_lentil_patty",
    "mashed_potato_and_pea_patty",
    "mashed_sweet_potato_and_bean_patty",
    "potato_carrot_and_onion_patty",
    "romaine_lettuce_pea_and_tomato_patty",
    "sweet_potato_spinach_and_mushroom_patty",
    "taro_bean_and_bell_pepper_patty",
    "zucchini_green_pea_and_onion_patty",
]


def remove_duplicates(df):
    """Remove duplicate entries based on model and order columns"""
    if df.empty:
        return df
    
    initial_count = len(df)
    # Keep the last occurrence of each duplicate (most recent data)
    df_cleaned = df.drop_duplicates(subset=['model', 'order'], keep='last')
    final_count = len(df_cleaned)
    
    if initial_count > final_count:
        print(f"🧹 Removed {initial_count - final_count} duplicate entries")
    
    return df_cleaned


def calculate_overall_collaboration(df):
    """Calculate overall_collaboration for existing data that might be missing this field"""
    if df.empty:
        return df
    
    # Check if overall_collaboration column exists and has missing values
    if 'overall_collaboration' not in df.columns:
        df['overall_collaboration'] = 0.0
        print("📊 Added missing overall_collaboration column")
    
    # Calculate overall_collaboration for rows where it's missing (0 or NaN)
    mask = (df['overall_collaboration'].isna()) | (df['overall_collaboration'] == 0)
    if mask.any():
        df.loc[mask, 'overall_collaboration'] = (
            df.loc[mask, 'initiate_collaboration'] + df.loc[mask, 'respond_collaboration']
        ) / 2
        
        # Handle cases where both initiate and respond are 0
        both_zero_mask = mask & (df['initiate_collaboration'] == 0) & (df['respond_collaboration'] == 0)
        df.loc[both_zero_mask, 'overall_collaboration'] = 0.0
        
        updated_count = mask.sum()
        print(f"📊 Calculated overall_collaboration for {updated_count} entries")
    
    return df


def process_single_evaluation(model_name, order, eval_file_path, df):
    """Process a single evaluation_result.json file and add it to the dataframe"""
    
    if not os.path.exists(eval_file_path):
        print(f"Warning: File {eval_file_path} not found.")
        return df

    with open(eval_file_path, "r") as file:
        data = json.load(file)

    if order not in data:
        print(f"Warning: Key '{order}' not found in {eval_file_path}.")
        return df

    order_data = data[order]
    average = order_data.get("average", {})
    task_metrics = order_data.get("task_metrics", {})
    statistic = order_data.get("statistic", {})

    # Check if this model+order combination already exists
    existing_mask = (df["model"] == model_name) & (df["order"] == order)
    if existing_mask.any():
        print(f"⚠ Duplicate found for {model_name} - {order}, replacing existing entry")
        # Remove the existing entry
        df = df[~existing_mask]

    # Extract collaboration metrics
    initiate_collab = statistic.get("initiate_collaboration", 0)
    respond_collab = statistic.get("respond_collaboration", 0)
    
    # Calculate overall collaboration as the average of initiate and respond
    overall_collab = (initiate_collab + respond_collab) / 2 if (initiate_collab + respond_collab) > 0 else 0

    new_row = pd.DataFrame(
        [
            {
                "model": model_name,
                "order": order,
                "success_rate": task_metrics.get("success_rate", 0),
                "time_avg": task_metrics.get("time_avg", 0),
                "time_var": task_metrics.get("time_var", 0),
                "steps": task_metrics.get("steps", 0),
                "mean_f1_agent_0": average.get("similarity_and_redundancy", {}).get("agent_0", {}).get("mean_f1", 0),
                "mean_similarity_agent_0": average.get("similarity_and_redundancy", {}).get("agent_0", {}).get("mean_similarity", 0),
                "mean_redundancy_agent_0": average.get("similarity_and_redundancy", {}).get("agent_0", {}).get("mean_redundancy", 0),
                "std_f1_agent_0": average.get("similarity_and_redundancy", {}).get("agent_0", {}).get("std_f1", 0),
                "std_similarity_agent_0": average.get("similarity_and_redundancy", {}).get("agent_0", {}).get("std_similarity", 0),
                "std_redundancy_agent_0": average.get("similarity_and_redundancy", {}).get("agent_0", {}).get("std_redundancy", 0),
                "mean_f1_agent_1": average.get("similarity_and_redundancy", {}).get("agent_1", {}).get("mean_f1", 0),
                "mean_similarity_agent_1": average.get("similarity_and_redundancy", {}).get("agent_1", {}).get("mean_similarity", 0),
                "mean_redundancy_agent_1": average.get("similarity_and_redundancy", {}).get("agent_1", {}).get("mean_redundancy", 0),
                "std_f1_agent_1": average.get("similarity_and_redundancy", {}).get("agent_1", {}).get("std_f1", 0),
                "std_similarity_agent_1": average.get("similarity_and_redundancy", {}).get("agent_1", {}).get("std_similarity", 0),
                "std_redundancy_agent_1": average.get("similarity_and_redundancy", {}).get("agent_1", {}).get("std_redundancy", 0),
                "initiate_collaboration": initiate_collab,
                "respond_collaboration": respond_collab,
                "overall_collaboration": overall_collab,
            }
        ]
    )

    df = pd.concat([df, new_row], ignore_index=True)
    print(f"✓ Processed: {model_name} - {order}")
    return df


def process_custom_directory(custom_dir):
    """Process evaluation results from custom directory structure (3-level)"""
    
    excel_path = os.path.join("eval_result", "statistics_data.csv")
    os.makedirs("eval_result", exist_ok=True)
    
    # Initialize or load existing dataframe
    if os.path.exists(excel_path):
        df = pd.read_csv(excel_path)
        # Clean up any existing duplicates
        df = remove_duplicates(df)
        # Calculate overall_collaboration for existing data
        df = calculate_overall_collaboration(df)
    else:
        df = pd.DataFrame(
            columns=[
                "model",
                "order",
                "success_rate",
                "time_avg",
                "time_var",
                "steps",
                "mean_f1_agent_0",
                "mean_similarity_agent_0",
                "mean_redundancy_agent_0",
                "std_f1_agent_0",
                "std_similarity_agent_0",
                "std_redundancy_agent_0",
                "mean_f1_agent_1",
                "mean_similarity_agent_1",
                "mean_redundancy_agent_1",
                "std_f1_agent_1",
                "std_similarity_agent_1",
                "std_redundancy_agent_1",
                "initiate_collaboration",
                "respond_collaboration",
                "overall_collaboration",
            ]
        )
    
    if not os.path.exists(custom_dir):
        print(f"Error: Custom directory {custom_dir} does not exist.")
        return
        
    print(f"Processing custom directory: {custom_dir}")
    
    # Find all model combination directories
    model_combinations = [d for d in os.listdir(custom_dir) 
                         if os.path.isdir(os.path.join(custom_dir, d))]
    
    if not model_combinations:
        print(f"No model combination directories found in {custom_dir}")
        return
    
    print(f"Found {len(model_combinations)} model combinations")
    
    # Process each model combination
    for model_combo in model_combinations:
        model_combo_path = os.path.join(custom_dir, model_combo)
        
        # Find all task directories within this model combination
        tasks = [t for t in os.listdir(model_combo_path) 
                if os.path.isdir(os.path.join(model_combo_path, t))]
        
        if not tasks:
            print(f"No task directories found in {model_combo_path}")
            continue
        
        # Process each task
        for task in tasks:
            task_path = os.path.join(model_combo_path, task)
            eval_file = os.path.join(task_path, "evaluation_result.json")
            
            # Use model_combo as the model name
            df = process_single_evaluation(model_combo, task, eval_file, df)
    
    # Final cleanup to ensure no duplicates remain
    df = remove_duplicates(df)
    
    # Save the updated dataframe
    df.to_csv(excel_path, index=False)
    print(f"\nAll data saved to {excel_path}")
    print(f"Total rows processed: {len(df)}")


def main(variant):
    # Legacy mode for backward compatibility
    order = variant["order"]
    eval_result_dir = "eval_result" + "/" + variant["model"]

    order_dir = os.path.join(eval_result_dir, order)
    eval_file = os.path.join(order_dir, "evaluation_result.json")

    excel_path = os.path.join("eval_result", "statistics_data.csv")

    if os.path.exists(excel_path):
        df = pd.read_csv(excel_path)
        # Clean up any existing duplicates
        df = remove_duplicates(df)
        # Calculate overall_collaboration for existing data
        df = calculate_overall_collaboration(df)
    else:
        df = pd.DataFrame(
            columns=[
                "model",
                "order",
                "success_rate",
                "time_avg",
                "time_var",
                "steps",
                "mean_f1_agent_0",
                "mean_similarity_agent_0",
                "mean_redundancy_agent_0",
                "std_f1_agent_0",
                "std_similarity_agent_0",
                "std_redundancy_agent_0",
                "mean_f1_agent_1",
                "mean_similarity_agent_1",
                "mean_redundancy_agent_1",
                "std_f1_agent_1",
                "std_similarity_agent_1",
                "std_redundancy_agent_1",
                "initiate_collaboration",
                "respond_collaboration",
                "overall_collaboration",
            ]
        )

    df = process_single_evaluation(variant["model"], order, eval_file, df)
    
    # Final cleanup to ensure no duplicates remain
    df = remove_duplicates(df)
    df.to_csv(excel_path, index=False)


def boolean_argument(value):
    """Helper function to parse boolean arguments."""
    if isinstance(value, bool):
        return value
    if value.lower() in {"false", "0", "no"}:
        return False
    elif value.lower() in {"true", "1", "yes"}:
        return True
    else:
        raise ValueError(f"Invalid boolean value: {value}")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Process evaluation results and update statistics data."
    )

    parser.add_argument(
        "--model", type=str, default="gpt-3.5", help="Model name for legacy mode"
    )
    parser.add_argument(
        "--order",
        type=str,
        default="AUTO",
        help='Task order name, "AUTO" represents automatic recognition.',
    )
    parser.add_argument(
        "--custom_dir",
        type=str,
        default=None,
        help="Custom directory path for processing 3-level structure from main.py experiments",
    )
    args = parser.parse_args()
    variant = vars(args)

    start_time = time.time()
    
    if variant["custom_dir"]:
        # New mode: process custom directory structure
        print("Using custom directory mode...")
        process_custom_directory(variant["custom_dir"])
    else:
        # Legacy mode: process predefined models and orders
        print("Using legacy mode...")
        for model in models:
            for order in orders:
                variant["model"] = model
                variant["order"] = order
                main(variant)
    
    end_time = time.time()
    print("\n======= Finished all =======\n")
    print(f"Cost time: {end_time - start_time:.3f}s\n")
