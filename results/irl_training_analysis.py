import pickle
import matplotlib.pyplot as plt
import numpy as np


def load_training_log(category: str):
    # Open the pickle file and load the data
    # remember change this accordingly
    with open(category + 'training_log.pkl', 'rb') as f:
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

    # plot the weight of one feature
    ax.plot(iteration, theta[:, 5])

    # Adding X and Y labels
    ax.set_xlabel('Iteration')
    ax.set_ylabel(key)

    ax.grid(True)


def plot_training_process(category):
    # plot how the variables like human likeness. feature differences, etc. change
    training_log = load_training_log(category)
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


def plot_weight_comparison(categories):
    # plot the weights comparison for different driving styles
    weights = {}
    # features = ['speed', 'long_acc', 'lat_acc', 'long_jerk', 'thw_front', 'thw_rear', 'induced_deceleration']
    features = ['speed', 'long_acc', 'lat_acc', 'long_jerk', 'thw_front', 'thw_rear', 'd_centerline','l_deviation_rate','left_available','right_available']
    for category in categories:
        training_log = load_training_log(category)
        for key, value in training_log.items():
            if key == 'theta':
                # weights[category] = np.concatenate((value[-1][:6], [value[-1][7]]))
                weights[category] = value[-1][:]

    # plot bar chart
    x = np.arange(len(features))  # the label locations
    width = 0.2  # the width of the bars

    # Create bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 1.5 * width, weights['Normal'], width, label='Normal', color='red')
    rects2 = ax.bar(x - 0.5 * width, weights['Aggressive'], width, label='Aggressive', color='green')
    rects3 = ax.bar(x + 0.5 * width, weights['Cautious'], width, label='Cautious', color='blue')
    rects4 = ax.bar(x + 1.5 * width, weights['All'], width, label='All', color='purple')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Features')
    ax.set_ylabel('Weights')
    ax.set_title('Weights for different driving styles')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=10)
    ax.legend()
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='gray')
    # Show the plot
    plt.show()


def main():
    categories = ['Normal', 'Aggressive', 'Cautious', 'All']
    plot_weight_comparison(categories)
    plot_training_process(categories[0])


if __name__ == "__main__":
    main()
