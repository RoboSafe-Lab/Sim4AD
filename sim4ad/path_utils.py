import json
import pathlib
from typing import List


def get_base_dir():
    return str(pathlib.Path(__file__).parent.parent.absolute()) + '/'


def get_config_path(scenario_name):
    config_path = f"{get_base_dir()}/scenarios/configs/{scenario_name}.json"
    return config_path


def _common_elements_path():
    return f'{get_base_dir()}/sim4ad/common_elements.json'


def baseline_path(file_name):
    return f"{get_base_dir()}/baselines/{file_name}.pth"


# FUNCTIONS FOR SHARED VARIABLES IN common_elements.json
def check_common_property_equals(property_name, value):
    """Check if the property is equal to the value in the common_elements.json, the file that keeps track of the shared
    variables across the project."""
    with open(_common_elements_path()) as f:
        common_elements = json.load(f)

    return common_elements[property_name] == value


def get_common_property(property_name):
    """Get the property from the common_elements.json, the file that keeps track of the shared variables across the
    project."""
    with open(_common_elements_path()) as f:
        common_elements = json.load(f)

    return common_elements[property_name]


def write_common_property(property_name, value):
    """Write the property with the value in the common_elements.json, the file that keeps track of the shared
    variables across the project."""
    with open(_common_elements_path(), 'r') as f:
        common_elements = json.load(f)

    common_elements[property_name] = value

    with open(_common_elements_path(), 'w') as f:
        json.dump(common_elements, f, indent=4)


def get_path_to_automatum_scenario(scenario_name):
    return f"{get_base_dir()}/scenarios/data/automatum/{scenario_name}"


def get_path_to_automatum_map(scenario_name):
    return f"{get_path_to_automatum_scenario(scenario_name)}/staticWorld.xodr"


def get_path_irl_weights(cluster):
    return f"{get_base_dir()}/results/{cluster}training_log.pkl"


def get_processed_demonstrations(split_type, scenario, cluster):
    return f"{get_base_dir()}/scenarios/data/{split_type}/{cluster}{scenario}_demonstration.pkl"


def get_file_name_trajectories(policy_type, spwan_method, irl_weights, episode_name: List[str], simulation_length):
    """
    Get the file name for the trajectories.

    :param policy_type: The policy type.
    :param irl_weights: The IRL weights.
    :param episode_name: The episode name.
    :return: The file name.
    """

    episode_name = "_".join(episode_name)
    if irl_weights is not None:
        irl_weights = "_".join([str(i) for i in irl_weights])

    folder_path = f"{get_base_dir()}/evaluation/trajectories/"
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)

    return f"{folder_path}/{episode_name}_{policy_type}_{spwan_method}_{irl_weights}_{simulation_length}steps.pkl"

def get_file_name_evaluation(policy_type, spawn_method, irl_weights, episode_name):
    """
    Get the file name for the evaluation.

    :param policy_type: The policy type.
    :param irl_weights: The IRL weights.
    :param episode_name: The episode nameS.
    :return: The file name.
    """
    episode_name = "_".join(episode_name)
    if irl_weights is not None:
        irl_weights = "_".join([str(i) for i in irl_weights])

    folder_path = f"{get_base_dir()}/evaluation/results/"
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)

    filename = f"{episode_name}_{policy_type}_{spawn_method}_{irl_weights}steps"
    return f"{folder_path}{filename}.pkl", filename

def get_agent_id_combined(episode_name, agent_id):
    return f"{episode_name}/{agent_id}"