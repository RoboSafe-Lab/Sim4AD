import json
import pathlib


def get_base_dir():
    return str(pathlib.Path(__file__).parent.parent.absolute()) + '/'


def get_config_path(scenario_name):
    config_path = f"{get_base_dir()}scenarios/configs/{scenario_name}.json"
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
