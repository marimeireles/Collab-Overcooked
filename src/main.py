import json
import os
import time
import warnings
from argparse import ArgumentParser
from distutils.util import strtobool

import importlib_metadata
from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from rich import print as rprint

import config
from collab.modules import statistics_dict
from collab.web_util import change_port, check_port_in_use, output_to_port
from utils import combine_statistic_dict, make_agent

# Get current working directory for paths
cwd = os.getcwd()
PROMPT_DIR = os.path.join(cwd, "prompts")


def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))


def is_openai_model(model_name):
    """
    Check if the given model name corresponds to an OpenAI model.
    Returns True for OpenAI models (gpt-*, openai models), False otherwise.
    """
    if not model_name:
        return False
    
    model_name_lower = model_name.lower()
    openai_indicators = ["gpt-", "openai", "chatgpt", "davinci", "curie", "babbage", "ada"]
    
    return any(indicator in model_name_lower for indicator in openai_indicators)


def check_recipe_parse(variant):
    """
    Verify that a recipe file matching variant['order'] exists under PROMPT_DIR/recipe/.
    Raise ValueError if not found.
    """
    recipe_name_list = os.listdir(PROMPT_DIR + "/recipe/")
    for r in recipe_name_list:
        if variant["order"] in r.lower():
            return True
    raise ValueError("Not valid order name!")


def main(variant):
    layout = variant["layout"]
    horizon = variant["horizon"]
    episode = variant["episode"]
    mode = variant["mode"]

    mdp = OvercookedGridworld.from_layout_name(layout)

    if variant["order"]:
        if check_recipe_parse(variant):
            mdp.start_order_list = [variant["order"]]
            mdp.one_task_mode = True

    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()

    print(f"\n===P0 agent: {variant['p0']} | P1 agent: {variant['p1']}===\n")

    start_time = time.time()
    results = []

    actor_list = ["chef", "assistant"]

    for i in range(episode):
        # Directory and filename for saving statistics
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = f"{variant['statistics_save_dir']}/{variant['gpt_model']}/{variant['order']}"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/experiment_{current_time}_{variant['order']}.json"

        # Develop mode: user steps through action_list manually
        if mode == "develop":
            action_list = []
            parm = []

            env.reset()
            r_total = 0
            for t in range(horizon):
                s_t = env.state
                print(f"\n>>>>>>>>>>>>>time: {t}<<<<<<<<<<<<<<<<<<<<<\n")
                print(env.mdp.state_string(s_t).replace("ø", "o"))

                obs, reward, done, env_info = env.step(action_list[t], parm[t])
                print(env.mdp.get_utensil_states(s_t))
                ml_actions = obs.ml_actions
                skills = ""
                for idx, ml_action in enumerate(ml_actions):
                    if ml_action is None:
                        continue
                    skills += f"P{idx} finished <{ml_action}>. "
                print(skills)

                r_total += reward
                rprint("[red]" + f"r: {reward} | total: {r_total}\n\n")
            # Exit after first develop run
            break

        # Build both agents
        agents_list = []
        for actor_idx, alg in enumerate([variant["p0"], variant["p1"]]):
            if alg == "LLMPair":
                # Validate that a GPT model is provided
                if mode != "human" and not variant["gpt_model"]:
                    raise ValueError(
                        "You must specify a model using --gpt_model when not in human mode"
                    )

                if mode == "OpenSource":
                    if not os.path.exists(variant["model_dirname"]):
                        raise ValueError(
                            f"Model directory not found: {variant['model_dirname']}"
                        )
                    print(f"Using open source model from: {variant['model_dirname']}")

                if variant["gpt_model"] == "human":
                    if not check_port_in_use(variant["local_server_api"]):
                        raise ValueError(
                            f"Port {variant['local_server_api']} is not in use"
                        )
                    change_port(variant["local_server_api"])
                    print("Running in human mode with local server")

                print(f"\n----Using model: {variant['gpt_model']}----\n")
                agent = make_agent(
                    alg,
                    mdp,
                    layout,
                    model=variant["gpt_model"],
                    model_dirname=variant["model_dirname"],
                    local_server_api=variant["local_server_api"],
                    retrival_method=variant["retrival_method"],
                    K=variant["K"],
                    actor=actor_list[actor_idx],
                )
            else:
                agent = make_agent(alg, mdp, layout)
            agents_list.append(agent)

        team = AgentGroup(*agents_list)
        team.reset()

        env.reset()
        r_total = 0

        # Experimental mode: Run the full game simulation
        if mode == "exp":
            # Main game loop - iterate through each time step
            for time_step in range(horizon):
                # Get current game state
                current_state = env.state
                
                print(f"\n>>>>>>>>>>>>>time: {time_step}<<<<<<<<<<<<<<<<<<<<<\n")
                
                # Convert and display the current map state (replace special characters)
                map_string = env.mdp.state_string(current_state).replace("ø", "o")
                print(map_string)

                # Check if Player 0 is using an OpenAI model and manage config accordingly
                p0_is_openai = (variant["p0"] == "LLMPair" and 
                               is_openai_model(variant["gpt_model"]))
                
                # Set OpenAI config flag if Player 0 uses OpenAI model
                if p0_is_openai:
                    config.cfg.set("settings", "openai_enabled", str(True))
                    config.save()
                    print('☀️B☀️E☀️G☀️I☀️N')
                try:
                    # Get joint action from both agents and any ingredient pickup parameters
                    joint_action, ingredient_for_pickup = team.joint_action(current_state)
                    print(joint_action)
                finally:
                    # Always reset OpenAI config flag back to False after Player 0's action
                    if p0_is_openai:
                        config.cfg.set("settings", "openai_enabled", str(False))
                        config.save()
                        print('☀️E☀️N☀️D')
                
                # Reset and get dialogue between agents
                dialogue_turn = team.reset_dialogue()
                
                print(f"\n-----------Controller-----------\n")
                print(
                    f"action: P0 {Action.to_char(joint_action[0])} | P1 {Action.to_char(joint_action[1])}"
                )
                
                # Set pickup parameters for the environment step
                action_parameters = ingredient_for_pickup

                # Execute the joint action in the environment
                observation, reward, done, env_info = env.step(joint_action, action_parameters)

                # Process and display completed machine learning actions (skills)
                ml_actions = observation.ml_actions
                completed_skills = ""
                for player_idx, ml_action in enumerate(ml_actions):
                    if ml_action is None:
                        continue
                    completed_skills += f"P{player_idx} finished <{ml_action}>. "
                print(completed_skills)

                # Update total reward
                r_total += reward
                
                # Handle successful order completion (positive reward)
                if reward > 0:
                    # Record the completed order in statistics
                    statistics_dict["total_order_finished"].append(
                        current_state.current_k_order[0]
                    )
                    # Log delivery action for agent 1's teammate tracking
                    team.agents[1].teammate_ml_actions.append(
                        {"timestamp": time_step, "action": "deliver_soup()"}
                    )

                # Display reward information with color formatting
                rprint("[red]" + f"r: {reward} | total: {r_total}\n\n")
                
                # Display agent behavior tracking
                print(f"P0's real behavior: {team.agents[1].teammate_ml_actions}")
                print(f"P1's real behavior: {team.agents[0].teammate_ml_actions}")

                # Collect per-turn statistics from both agents
                turn_statistics_agent0 = team.agents[0].turn_statistics_dict
                turn_statistics_agent1 = team.agents[1].turn_statistics_dict

                # Combine statistics from both agents with environment data
                combined_turn_statistics = combine_statistic_dict(
                    turn_statistics_agent0,
                    turn_statistics_agent1,
                    map_string,
                    reward,
                )

                # Update global statistics dictionary
                statistics_dict["total_timestamp"].append(time_step)
                statistics_dict["total_score"] = r_total
                # Note: Agent indices are swapped for teammate action tracking
                statistics_dict["total_action_list"][0] = team.agents[1].teammate_ml_actions
                statistics_dict["total_action_list"][1] = team.agents[0].teammate_ml_actions
                statistics_dict["content"].append(combined_turn_statistics)

                # Save statistics to file after each turn
                with open(filename, "w") as statistics_file:
                    json.dump(statistics_dict, statistics_file, indent=4)

                # Check for task completion in fixed task mode
                if variant["test_mode"] == "fix_task" and reward != 0:
                    print("Task succeeded!")
                    if variant["gpt_model"] == "human":
                        for a in range(len(team.agents)):
                            output_to_port(
                                f"agent{a}",
                                "Success!",
                                mission="success",
                                port=variant["local_server_api"],
                            )
                    break

            if variant["gpt_model"] == "human":
                for a in range(len(team.agents)):
                    output_to_port(
                        f"agent{a}",
                        "Fail to finish task in time!",
                        mission="fail",
                        port=variant["local_server_api"],
                    )

        print(f"Episode {i + 1}/{episode}: {r_total}\n====\n\n")
        results.append(r_total)

    end_time = time.time()
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="OvercookedAI Experiment")

    parser.add_argument(
        "--layout", "-l", type=str, default="new_env", choices=["new_env"]
    )
    parser.add_argument(
        "--p0",
        type=str,
        default="LLMPair",
        choices=["LLMPair", "Human"],
        help="Algorithm for P0 agent",
    )
    parser.add_argument(
        "--p1",
        type=str,
        default="LLMPair",
        choices=["LLMPair", "Human"],
        help="Algorithm for P1 agent",
    )
    parser.add_argument(
        "--horizon", type=int, default=120, help="Horizon steps in one game"
    )
    parser.add_argument("--episode", type=int, default=1, help="Number of episodes")
    parser.add_argument(
        "--gpt_model",
        type=str,
        default="gpt-3.5-turbo-0125",
        help="LLM model (e.g. gpt-4, llama3-8B)",
    )
    parser.add_argument(
        "--retrival_method",
        type=str,
        default="recent_k",
        choices=["recent_k", "bert_topk"],
        help="Retrieval method for dialogue history",
    )
    parser.add_argument(
        "--K", type=int, default=0, help="Number of dialogues to retrieve"
    )
    parser.add_argument(
        "--model_dirname",
        type=str,
        default=".",
        help="Absolute path of open-source model directory",
    )
    parser.add_argument(
        "--local_server_api",
        type=str,
        default="http://localhost:8000/v1",
        help="URL for local LLM server",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="exp",
        choices=["exp", "debug_validator", "develop"],
        help="exp mode run step-by-step, demo mode run via traj",
    )
    parser.add_argument(
        "--test_mode", type=str, default="fix_task", choices=["fix_task", "fix_time"]
    )
    parser.add_argument(
        "--save", type=boolean_argument, default=True, help="Whether save the result"
    )
    parser.add_argument("--log_dir", type=str, default=None, help="dir to save result")
    parser.add_argument(
        "--debug", type=boolean_argument, default=True, help="debug mode"
    )
    parser.add_argument("--order", type=str, default="", help="1 task order name")
    parser.add_argument(
        "--statistics_save_dir",
        type=str,
        default="data",
        help="save directory of LLM statistics",
    )
    # parser.add_argument(
    #     "--openai",
    #     type=boolean_argument,
    #     default=False,
    #     help="Enable OpenAI API usage",
    # )

    args = parser.parse_args()
    variant = vars(args)
    # config.cfg.set("settings", "openai_enabled", str(args.openai))
    # config.save()

    start_time = time.time()
    main(variant)
    end_time = time.time()
    print(f"\n=======Finshed all=========\n")
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")
