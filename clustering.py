from util import parse_args
import sys

from sim4ad.data.data_loaders import DatasetDataLoader


def main():
    args = parse_args()
    data_loader = DatasetDataLoader(f"scenarios/configs/{args.map}.json")
    data_loader.load()
    print('ok')


if __name__ == '__main__':
    sys.exit(main())
