from .data_loaders import DataLoader, DatasetDataLoader
from .episode import Episode, EpisodeConfig, EpisodeMetadata, EpisodeLoader, DatasetEpisodeLoader
from .scenario import Scenario, ScenarioConfig, DatasetScenario

EpisodeLoader.register_loader("automatum", DatasetEpisodeLoader)