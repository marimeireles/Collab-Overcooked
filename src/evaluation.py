import datetime
import json
import os
import re
import time
from argparse import ArgumentParser
from distutils.util import strtobool

import pandas as pd
from rich import print as rprint

from eval_utils import Evaluation, ExpLog

models = ["gpt-4o"]

orders = [
    "apple",
    "carrot",
    "onion",
    "potato",
    "tofu",
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


def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))


def main(variant):

    if variant["mode"] == "exp":

        if variant["test_mode"] == "fix_task":
            auto_order_list = []
            if variant["order"] == "AUTO":
                exp_log = ExpLog(variant["log_dir"] + "/" + variant["model"])
                for idx in range(exp_log.__len__()):
                    auto_order_list.append(exp_log.get_secondary_order_list(idx, 0)[0])
                eval = Evaluation(order_name_list=auto_order_list, exp_log=exp_log)
            else:
                exp_log = ExpLog(
                    variant["log_dir"] + "/" + variant["model"] + "/" + variant["order"]
                )
                for idx in range(exp_log.__len__()):
                    auto_order_list.append(variant["order"])
                eval = Evaluation(order_name_list=auto_order_list, exp_log=exp_log)

        if variant["test_mode"] == "custom_dir":
            # New mode: handles 3-level structure (experiment_date/model_combination/task)
            # Discover all model combinations and tasks automatically
            base_dir = variant["log_dir"]
            
            if not os.path.exists(base_dir):
                print(f"Error: Directory {base_dir} does not exist.")
                return
            
            # Find all model combination directories
            model_combinations = [d for d in os.listdir(base_dir) 
                                if os.path.isdir(os.path.join(base_dir, d))]
            
            if not model_combinations:
                print(f"No model combination directories found in {base_dir}")
                return
            
            print(f"Found {len(model_combinations)} model combinations:")
            for combo in model_combinations:
                print(f"  - {combo}")
            
            # Process each model combination
            for i, model_combo in enumerate(model_combinations):
                print(f"\n[{i+1}/{len(model_combinations)}] Processing model combination: {model_combo}")
                model_combo_path = os.path.join(base_dir, model_combo)
                
                # Find all task directories within this model combination
                tasks = [t for t in os.listdir(model_combo_path) 
                        if os.path.isdir(os.path.join(model_combo_path, t))]
                
                if not tasks:
                    print(f"No task directories found in {model_combo_path}")
                    continue
                
                print(f"\nProcessing {model_combo} with tasks: {tasks}")
                
                # Process each task
                for task in tasks:
                    task_path = os.path.join(model_combo_path, task)
                    print(f"    Processing task: {task} in {task_path}")
                    
                    try:
                        # Create ExpLog for this specific task directory
                        exp_log = ExpLog(task_path)
                        
                        if exp_log.__len__() == 0:
                            print(f"    ⚠️  No experiment files found in {task_path}")
                            continue
                        
                        print(f"    Found {exp_log.__len__()} experiment files")
                        
                        # Create order list for this task
                        auto_order_list = []
                        if variant["order"] == "AUTO":
                            # Auto-detect orders from JSON files
                            for idx in range(exp_log.__len__()):
                                try:
                                    detected_order = exp_log.get_secondary_order_list(idx, 0)[0]
                                    auto_order_list.append(detected_order)
                                except:
                                    # If auto-detection fails, use the directory name as task
                                    auto_order_list.append(task)
                        else:
                            # Use specified order for all logs
                            for idx in range(exp_log.__len__()):
                                auto_order_list.append(variant["order"])
                        
                        print(f"    Order list: {auto_order_list}")
                        
                        # Create evaluation and run it
                        eval = Evaluation(order_name_list=auto_order_list, exp_log=exp_log)
                        eval.evaluate(task_path)
                        
                        print(f"    ✓ Completed evaluation for {model_combo}/{task}")
                        
                    except Exception as e:
                        print(f"    ✗ Error evaluating {model_combo}/{task}: {str(e)}")
                        import traceback
                        print(f"    Full error: {traceback.format_exc()}")
                        continue
            
            print(f"\nCustom directory evaluation completed for {variant['log_dir']}")

        if variant["test_mode"] == "build_in":
            for model in models:
                for order in orders:
                    auto_order_list = []
                    variant["model"] = model
                    variant["order"] = order
                    exp_log = ExpLog(
                        variant["log_dir"]
                        + "/"
                        + variant["model"]
                        + "/"
                        + variant["order"]
                    )
                    for idx in range(exp_log.__len__()):
                        auto_order_list.append(variant["order"])
                    eval = Evaluation(order_name_list=auto_order_list, exp_log=exp_log)

                    # print(eval.evaluate(variant['save_dir']))
                    eval.evaluate(
                        variant["log_dir"]
                        + "/"
                        + variant["model"]
                        + "/"
                        + variant["order"]
                    )


if __name__ == "__main__":

    parser = ArgumentParser(description="OvercookedAI Experiment")

    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo-0125", help="Number of episodes"
    )

    parser.add_argument(
        "--K", type=int, default=0, help="The number of dialogues you want to retrieve."
    )

    #
    parser.add_argument(
        "--mode",
        type=str,
        default="exp",
        choices=["exp", "develop"],
        help="exp mode run step-by-step, demo mode run via traj",
    )
    parser.add_argument(
        "--test_mode", type=str, default="fix_task", choices=["fix_task", "build_in", "custom_dir"]
    )
    parser.add_argument(
        "--save", type=boolean_argument, default=True, help="Whether save the result"
    )
    parser.add_argument(
        "--log_dir", type=str, default="eval_result", help="dir to save result"
    )
    parser.add_argument(
        "--debug", type=boolean_argument, default=True, help="debug mode"
    )
    parser.add_argument(
        "--order",
        type=str,
        default="AUTO",
        help='1 task order name, "AUTO" represents automatic recognition',
    )
    parser.add_argument(
        "--recipe_dir",
        type=str,
        default="prompts/recipe",
        help="The dir of the recipe and reference",
    )

    #
    parser.add_argument(
        "--save_dir",
        type=str,
        default="eval_result",
        help="save directory of LLM statistics",
    )

    args = parser.parse_args()
    variant = vars(args)

    start_time = time.time()
    main(variant)
    end_time = time.time()
    print(f"\n=======Finshed all=========\n")
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")
