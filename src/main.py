import os
import json
import time
from argparse import ArgumentParser
from distutils.util import strtobool
import warnings
import importlib_metadata

from rich import print as rprint

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.actions import Action

from collab.modules import statistics_dict
from collab.web_util import output_to_port, check_port_in_use, change_port
from utils import make_agent, combine_statistic_dict


def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))


def check_recipe_parse(variant):
    """
    Verify that a recipe file matching variant['order'] exists under PROMPT_DIR/recipe/.
    Raise ValueError if not found.
    """
    recipe_name_list = os.listdir(PROMPT_DIR + '/recipe/')
    for r in recipe_name_list:
        if variant['order'] in r.lower():
            return True
    raise ValueError("Not valid order name!")


def main(variant):
    layout = variant['layout']
    horizon = variant['horizon']
    episode = variant['episode']
    mode = variant['mode']

    mdp = OvercookedGridworld.from_layout_name(layout)

    if variant['order']:
        if check_recipe_parse(variant):
            mdp.start_order_list = [variant['order']]
            mdp.one_task_mode = True

    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()

    print(f"\n===P0 agent: {variant['p0']} | P1 agent: {variant['p1']}===\n")

    start_time = time.time()
    results = []

    actor_list = ['chef', 'assistant']

    for i in range(episode):
        # Directory and filename for saving statistics
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = f"{variant['statistics_save_dir']}/{variant['gpt_model']}/{variant['order']}"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/experiment_{current_time}_{variant['order']}.json"

        # Develop mode: user steps through action_list manually
        if mode == 'develop':
            action_list = []
            parm = []

            env.reset()
            r_total = 0
            for t in range(horizon):
                s_t = env.state
                print(f'\n>>>>>>>>>>>>>time: {t}<<<<<<<<<<<<<<<<<<<<<\n')
                print(env.mdp.state_string(s_t).replace('ø', 'o'))

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
                rprint("[red]" + f'r: {reward} | total: {r_total}\n\n')
            # Exit after first develop run
            break

        # Build both agents
        agents_list = []
        for actor_idx, alg in enumerate([variant['p0'], variant['p1']]):
            if alg == "LLMPair":
                # Validate that a GPT model is provided
                if mode != "human" and not variant['gpt_model']:
                    raise ValueError("You must specify a model using --gpt_model when not in human mode")

                if mode == "OpenSource":
                    if not os.path.exists(variant['model_dirname']):
                        raise ValueError(f"Model directory not found: {variant['model_dirname']}")
                    print(f"Using open source model from: {variant['model_dirname']}")

                if variant['gpt_model'] == "human":
                    if not check_port_in_use(variant["local_server_api"]):
                        raise ValueError(f"Port {variant['local_server_api']} is not in use")
                    change_port(variant["local_server_api"])
                    print("Running in human mode with local server")

                print(f"\n----Using model: {variant['gpt_model']}----\n")
                agent = make_agent(
                    alg,
                    mdp,
                    layout,
                    model=variant['gpt_model'],
                    model_dirname=variant['model_dirname'],
                    local_server_api=variant['local_server_api'],
                    retrival_method=variant['retrival_method'],
                    K=variant['K'],
                    actor=actor_list[actor_idx]
                )
            else:
                agent = make_agent(alg, mdp, layout)
            agents_list.append(agent)

        team = AgentGroup(*agents_list)
        team.reset()

        env.reset()
        r_total = 0

        if mode == 'exp':
            for t in range(horizon):
                s_t = env.state
                print(f'\n>>>>>>>>>>>>>time: {t}<<<<<<<<<<<<<<<<<<<<<\n')
                map_str = env.mdp.state_string(s_t).replace('ø', 'o')
                print(map_str)

                a_t, ingredient_for_pickup = team.joint_action(s_t)
                print(a_t)
                dialogue_t = team.reset_dialogue()
                print(f"\n-----------Controller-----------\n")
                print(f"action: P0 {Action.to_char(a_t[0])} | P1 {Action.to_char(a_t[1])}")
                parm = ingredient_for_pickup

                obs, reward, done, env_info = env.step(a_t, parm)

                ml_actions = obs.ml_actions
                skills = ""
                for idx, ml_action in enumerate(ml_actions):
                    if ml_action is None:
                        continue
                    skills += f"P{idx} finished <{ml_action}>. "
                print(skills)

                r_total += reward
                if reward > 0:
                    statistics_dict['total_order_finished'].append(s_t.current_k_order[0])
                    team.agents[1].teammate_ml_actions.append({'timestamp': t, 'action': "deliver_soup()"})

                rprint("[red]" + f'r: {reward} | total: {r_total}\n\n')
                print(f"P0's real behavior: {team.agents[1].teammate_ml_actions}")
                print(f"P1's real behavior: {team.agents[0].teammate_ml_actions}")

                # Save per-turn statistics
                turn_statistics_dict_agent0 = team.agents[0].turn_statistics_dict
                turn_statistics_dict_agent1 = team.agents[1].turn_statistics_dict

                turn_statistics_dict_both = combine_statistic_dict(
                    turn_statistics_dict_agent0,
                    turn_statistics_dict_agent1,
                    map_str,
                    reward
                )

                statistics_dict['total_timestamp'].append(t)
                statistics_dict['total_score'] = r_total
                statistics_dict['total_action_list'][0] = team.agents[1].teammate_ml_actions
                statistics_dict['total_action_list'][1] = team.agents[0].teammate_ml_actions
                statistics_dict['content'].append(turn_statistics_dict_both)

                with open(filename, 'w') as f:
                    json.dump(statistics_dict, f, indent=4)

                if variant['test_mode'] == 'fix_task' and reward != 0:
                    print("Task succeeded!")
                    if variant['gpt_model'] == "human":
                        for a in range(len(team.agents)):
                            output_to_port(
                                f"agent{a}",
                                "Success!",
                                mission="success",
                                port=variant['local_server_api']
                            )
                    break

            if variant['gpt_model'] == "human":
                for a in range(len(team.agents)):
                    output_to_port(
                        f"agent{a}",
                        "Fail to finish task in time!",
                        mission="fail",
                        port=variant['local_server_api']
                    )

        print(f"Episode {i + 1}/{episode}: {r_total}\n====\n\n")
        results.append(r_total)

    end_time = time.time()
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")


if __name__ == '__main__':
    parser = ArgumentParser(description='OvercookedAI Experiment')

    parser.add_argument('--layout', '-l', type=str, default='new_env', choices=['new_env'])
    parser.add_argument('--p0', type=str, default='LLMPair', choices=['LLMPair', 'Human'],
                        help='Algorithm for P0 agent')
    parser.add_argument('--p1', type=str, default='LLMPair', choices=['LLMPair', 'Human'],
                        help='Algorithm for P1 agent')
    parser.add_argument('--horizon', type=int, default=120, help='Horizon steps in one game')
    parser.add_argument('--episode', type=int, default=1, help='Number of episodes')
    parser.add_argument('--gpt_model', type=str, default='gpt-3.5-turbo-0125',
                        help='LLM model (e.g. gpt-4, llama3-8B)')
    parser.add_argument('--retrival_method', type=str, default="recent_k",
                        choices=['recent_k', 'bert_topk'],
                        help='Retrieval method for dialogue history')
    parser.add_argument('--K', type=int, default=0,
                        help='Number of dialogues to retrieve')
    parser.add_argument('--model_dirname', type=str, default='.',
                        help='Absolute path of open-source model directory')
    parser.add_argument('--local_server_api', type=str, default="http://localhost:8000/v1",
                        help='URL for local LLM server')
    parser.add_argument('--mode', type=str, default='exp', choices=['exp', 'debug_validator', 'develop'], help='exp mode run step-by-step, demo mode run via traj')                                
    parser.add_argument('--test_mode', type=str, default='fix_task', choices=['fix_task', 'fix_time'])
    parser.add_argument('--save', type=boolean_argument, default=True, help='Whether save the result')
    parser.add_argument('--log_dir', type=str, default=None, help='dir to save result')
    parser.add_argument('--debug', type=boolean_argument, default=True, help='debug mode')
    parser.add_argument('--order', type=str, default="", help='1 task order name')

    parser.add_argument('--statistics_save_dir', type=str, default='data', help='save directory of LLM statistics')


    args = parser.parse_args()
    variant = vars(args)

    start_time = time.time()
    main(variant)
    end_time = time.time()
    print(f"\n=======Finshed all=========\n")
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")
