import gymnasium as gym
import numpy as np
from tqdm import tqdm

from simulator.lightweight_simulator import Sim4ADSimulation
from baselines.sac.model import Actor as SACActor
import torch

ep_name = ["hw-a9-appershofen-001-d8087340-8287-46b6-9612-869b09e68448",
               "hw-a9-appershofen-002-2234a9ae-2de1-4ad4-9f43-65c2be9696d6"]

spawn_method = "dataset_all"
# "bc-all-obs-5_pi_cluster_Aggressive"  # "bc-all-obs-1.5_pi" "idm"
policy_type = "sac"  # "follow_dataset"
clustering = "all"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("SimulatorEnv-v0", dataset_split="test")
model = SACActor(env, device=device).to(device)
model.load_state_dict(torch.load("best_model_sac_SimulatorEnv-v0__model__1__1721400747.pth"))

DRIVING_STYLES = {
    "Aggressive": model,
    "Normal": model,
    "Cautious": model
}

sim = Sim4ADSimulation(episode_name=ep_name, spawn_method=spawn_method, policy_type=policy_type,
                       clustering=clustering, driving_style_policies=DRIVING_STYLES)
sim.full_reset()

# done = False # TODO: uncomment this to run until we use all vehicles
# while not done:
#     assert spawn_method != "random", "we will never finish!"
#     done = sim.step(return_done=True)

simulation_length = 50  # seconds
for _ in tqdm(range(int(np.floor(simulation_length / sim.dt)))):
    sim.step()

# Remove all agents left in the simulation.
sim.kill_all_agents()

sim.replay_simulation()

print("Simulation done!")
