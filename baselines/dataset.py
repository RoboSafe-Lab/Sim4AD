from torch.utils.data import Dataset


class AutomatumDataset(Dataset):
    def __init__(self, trajectories, actions):
        self.trajectories = trajectories
        self.actions = actions

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx], self.actions[idx]
