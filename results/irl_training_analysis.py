import pickle
import matplotlib.pyplot as plt
import numpy as np


def load_training_log():
    # Open the pickle file and load the data
    # remember change this accordingly
    with open('Aggressivetraining_log.pkl', 'rb') as f:
        training_log = pickle.load(f)

    return training_log


def plot_training_variable(iteration, key, value):
    # Create a figure and an axes
    fig, ax = plt.subplots()

    # Plotting the line
    ax.plot(iteration, value)

    # Adding X and Y labels
    ax.set_xlabel('Iteration')
    ax.set_ylabel(key)

    ax.grid(True)


def plot_theta(iteration, key, theta):
    theta = np.array(theta)
    """Plot the changing of rewards weights"""
    fig, ax = plt.subplots()

    # Plotting the line
    ax.plot(iteration, theta[:, 5])

    # Adding X and Y labels
    ax.set_xlabel('Iteration')
    ax.set_ylabel(key)

    ax.grid(True)


def main():
    training_log = load_training_log()
    length = len(training_log['iteration'])
    iteration = range(1, length + 1)
    for key, value in training_log.items():
        value = value[:length]
        if key != 'iteration' and key != 'theta':
            plot_training_variable(iteration, key, value)
        elif key == 'theta':
            plot_theta(iteration, key, value)

            # print theta at the last row
            print(f'The final theta are: {value[-1]}')
    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
