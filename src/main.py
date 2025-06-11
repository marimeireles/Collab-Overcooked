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

from collab.modules import statistics_dict
from collab.web_util import change_port, check_port_in_use, output_to_port
from utils import combine_statistic_dict, make_agent

# Get current working directory for paths
cwd = os.getcwd()
PROMPT_DIR = os.path.join(cwd, "prompts")


def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))



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

        # Build P0 agent (chef)
        p0_model = variant["p0_gpt_model"] or variant["gpt_model"]
        p0_model_dirname = variant["p0_model_dirname"] or variant["model_dirname"]
        p0_local_server_api = variant["p0_local_server_api"] or variant["local_server_api"]
        
        if variant["p0"] == "LLMPair":
            if mode != "human" and not p0_model:
                raise ValueError("You must specify a model for P0 using --p0_gpt_model or --gpt_model")
            if mode == "OpenSource" and not os.path.exists(p0_model_dirname):
                raise ValueError(f"P0 model directory not found: {p0_model_dirname}")
            if p0_model == "human":
                if not check_port_in_use(p0_local_server_api):
                    raise ValueError(f"P0 port {p0_local_server_api} is not in use")
                change_port(p0_local_server_api)
            print(f"\n----P0 using model: {p0_model}----\n")
            p0_agent = make_agent("LLMPair", mdp, layout,
                                model=p0_model, model_dirname=p0_model_dirname,
                                local_server_api=p0_local_server_api,
                                retrival_method=variant["retrival_method"],
                                K=variant["K"], actor="chef")
        else:
            p0_agent = make_agent(variant["p0"], mdp, layout)

        # Build P1 agent (assistant)
        p1_model = variant["p1_gpt_model"] or variant["gpt_model"]
        p1_model_dirname = variant["p1_model_dirname"] or variant["model_dirname"]
        p1_local_server_api = variant["p1_local_server_api"] or variant["local_server_api"]
        
        if variant["p1"] == "LLMPair":
            if mode != "human" and not p1_model:
                raise ValueError("You must specify a model for P1 using --p1_gpt_model or --gpt_model")
            if mode == "OpenSource" and not os.path.exists(p1_model_dirname):
                raise ValueError(f"P1 model directory not found: {p1_model_dirname}")
            if p1_model == "human":
                if not check_port_in_use(p1_local_server_api):
                    raise ValueError(f"P1 port {p1_local_server_api} is not in use")
                change_port(p1_local_server_api)
            print(f"\n----P1 using model: {p1_model}----\n")
            p1_agent = make_agent("LLMPair", mdp, layout,
                                model=p1_model, model_dirname=p1_model_dirname,
                                local_server_api=p1_local_server_api,
                                retrival_method=variant["retrival_method"],
                                K=variant["K"], actor="assistant")
        else:
            p1_agent = make_agent(variant["p1"], mdp, layout)

        agents_list = [p0_agent, p1_agent]

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
                
                print('☀️B☀️E☀️G☀️I☀️N')
                # Get joint action from both agents and any ingredient pickup parameters
                # TODO: crap, this comes from the lib, need to inspect tomorrow whether this
                # has an action taking function that's separated or if it's only this joint
                # thing. inspecting the code ive already learned it's separated actions and then
                # they join it as a tuple, so it's not going to be disgustingly difficult
                # just annoying to  transmit this informtion somehow....
                # I have no idea how actually
                joint_action, ingredient_for_pickup = team.joint_action(current_state)
                print(joint_action)
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
                    if p0_model == "human":
                        output_to_port("agent0", "Success!", mission="success", port=p0_local_server_api)
                    if p1_model == "human":
                        output_to_port("agent1", "Success!", mission="success", port=p1_local_server_api)
                    break

            if p0_model == "human":
                output_to_port("agent0", "Fail to finish task in time!", mission="fail", port=p0_local_server_api)
            if p1_model == "human":
                output_to_port("agent1", "Fail to finish task in time!", mission="fail", port=p1_local_server_api)

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
        help="LLM model (e.g. gpt-4, llama3-8B) - used when p0_gpt_model and p1_gpt_model are not specified",
    )
    parser.add_argument(
        "--p0_gpt_model",
        type=str,
        default=None,
        help="LLM model for P0 agent (overrides --gpt_model for P0)",
    )
    parser.add_argument(
        "--p1_gpt_model",
        type=str,
        default=None,
        help="LLM model for P1 agent (overrides --gpt_model for P1)",
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
        help="Absolute path of open-source model directory - used when p0_model_dirname and p1_model_dirname are not specified",
    )
    parser.add_argument(
        "--p0_model_dirname",
        type=str,
        default=None,
        help="Absolute path of open-source model directory for P0 agent (overrides --model_dirname for P0)",
    )
    parser.add_argument(
        "--p1_model_dirname",
        type=str,
        default=None,
        help="Absolute path of open-source model directory for P1 agent (overrides --model_dirname for P1)",
    )
    parser.add_argument(
        "--local_server_api",
        type=str,
        default="http://localhost:8000/v1",
        help="URL for local LLM server - used when p0_local_server_api and p1_local_server_api are not specified",
    )
    parser.add_argument(
        "--p0_local_server_api",
        type=str,
        default=None,
        help="URL for local LLM server for P0 agent (overrides --local_server_api for P0)",
    )
    parser.add_argument(
        "--p1_local_server_api",
        type=str,
        default=None,
        help="URL for local LLM server for P1 agent (overrides --local_server_api for P1)",
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

    args = parser.parse_args()
    variant = vars(args)

    start_time = time.time()
    main(variant)
    end_time = time.time()
    print(f"\n=======Finshed all=========\n")
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")
