from sim4ad.data.data_loaders import DatasetDataLoader
import math

def load_dataset():
    """Loading the dataset"""
    data_loader = DatasetDataLoader(f"scenarios/configs/automatum.json")
    data_loader.load()

    episodes = data_loader.scenario.episodes

    return episodes


# load the dataset
episodes = load_dataset()

for episode in episodes:
    for frame in episode.frames:
        if math.isclose(frame.time, 37.37, rel_tol=1e-2):
            for agent_key in frame.agents.keys():
                print(agent_key)
            break


pass
