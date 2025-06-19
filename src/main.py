import json
import os
import time
from argparse import ArgumentParser
from distutils.util import strtobool

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


def sanitize_model_name_for_path(model_name):
    """Convert model name to filesystem-safe string by replacing problematic characters."""
    if not model_name:
        return model_name
    # Replace forward slashes with underscores and other problematic characters
    return model_name.replace("/", "_").replace(":", "_").replace(" ", "_")

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


def build_agent(variant, key, actor, mdp, layout, mode):
    """Create an agent (P0 or P1) based on the provided configuration key.

    Parameters
    ----------
    variant : dict
        Parsed command-line arguments.
    key : str
        Either "p0" or "p1" - used to look up per-player flags.
    actor : str
        "chef" for P0, "assistant" for P1 - forwarded to make_agent.
    mdp, layout, mode
        Environment and mode information forwarded from main.
    """
    model = variant[f"{key}_gpt_model"] or variant["gpt_model"]
    model_dirname = variant[f"{key}_model_dirname"] or variant["model_dirname"]
    local_server_api = variant[f"{key}_local_server_api"] or variant["local_server_api"]
    algo = variant[key]

    if algo == "LLMPair":
        if mode != "human" and not model:
            raise ValueError(
                f"You must specify a model for {key.upper()} using --{key}_gpt_model or --gpt_model"
            )
        if mode == "OpenSource" and not os.path.exists(model_dirname):
            raise ValueError(f"{key.upper()} model directory not found: {model_dirname}")
        if model == "human":
            if not check_port_in_use(local_server_api):
                raise ValueError(f"{key.upper()} port {local_server_api} is not in use")
            change_port(local_server_api)
        print(f"\n----{key.upper()} using model: {model}----\n")
        return make_agent(
            "LLMPair",
            mdp,
            layout,
            model=model,
            model_dirname=model_dirname,
            local_server_api=local_server_api,
            retrival_method=variant["retrival_method"],
            K=variant["K"],
            actor=actor,
        )
    else:
        return make_agent(algo, mdp, layout)


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

    for i in range(episode):
        # Directory and filename for saving statistics
        # Determine effective model names for P0 and P1 (fallbacks to --gpt_model when not provided)
        p0_model = variant["p0_gpt_model"] or variant["gpt_model"]
        p1_model = variant["p1_gpt_model"] or variant["gpt_model"]
        
        # Sanitize model names for filesystem use
        p0_model_safe = sanitize_model_name_for_path(p0_model)
        p1_model_safe = sanitize_model_name_for_path(p1_model)

        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        # Save directory is now <statistics_save_dir>/<p0_model_safe>_<p1_model_safe>/<order>
        save_dir = f"{variant['statistics_save_dir']}/{p0_model_safe}_{p1_model_safe}/{variant['order']}"
        os.makedirs(save_dir, exist_ok=True)
        # Filename embeds model names for clarity as well
        filename = f"{save_dir}/experiment_{current_time}_chef_{p0_model_safe}_assistant_{p1_model_safe}_{variant['order']}.json"

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

        # Build agents (P0 – chef, P1 – assistant) with shared helper
        player_configs = [("p0", "chef"), ("p1", "assistant")]
        agents_list = [
            build_agent(variant, player_key, actor, mdp, layout, mode)
            for player_key, actor in player_configs
        ]

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
                
                # Get joint action from both agents and any ingredient pickup parameters
                joint_action, ingredient_for_pickup = team.joint_action(current_state)
                print(joint_action)

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
                        output_to_port("agent0", "Success!", mission="success", port=variant["p0_local_server_api"])
                    if p1_model == "human":
                        output_to_port("agent1", "Success!", mission="success", port=variant["p1_local_server_api"])
                    break

            if p0_model == "human":
                output_to_port("agent0", "Fail to finish task in time!", mission="fail", port=variant["p0_local_server_api"])
            if p1_model == "human":
                output_to_port("agent1", "Fail to finish task in time!", mission="fail", port=variant["p1_local_server_api"])

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
