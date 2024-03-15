import pickle
import matplotlib.pyplot as plt


def load_training_log():
    # Open the pickle file and load the data
    with open('training_log.pkl', 'rb') as f:
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


def main():
    training_log = load_training_log()
    iteration = training_log['iteration']
    for key, value in training_log.items():
        if key != 'iteration' and key != 'theta':
            plot_training_variable(iteration, key, value)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
