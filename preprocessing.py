import sys
from sim4ad.util import parse_args
from sim4ad.data.data_loaders import DatasetDataLoader
from clustering import FeatureExtraction, Clustering
from extract_observation_action import ExtractObservationAction


def main():
    args = parse_args()

    data_loader = DatasetDataLoader(f"scenarios/configs/{args.map}.json")
    data_loader.load()

    episodes = data_loader.scenario.episodes
    feature_extractor = FeatureExtraction(episodes)

    # extract feature values from the dataset
    df = feature_extractor.extract_features()
    # begin clustering
    cluster = Clustering()
    clustered_dataframe = cluster.GMM(df)

    # extract observations and actions
    extractor = ExtractObservationAction(episodes, clustered_dataframe)
    extractor.extract_demonstrations()
    extractor.save_trajectory()


if __name__ == '__main__':
    sys.exit(main())
