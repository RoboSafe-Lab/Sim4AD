import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    features = [r'$v_{\rm ego}$', r'$a_{\rm long}$', r'$a_{\rm lat}$', r'$J_{\rm long}$', r'$THW_{\rm f}$', r'$THW_{\rm r}$', r'$d_{\rm c}$', r'$\dot{d}_{\rm c}$', r'$avail_{\rm l}$', r'$avail_{\rm r}$']
    for category in categories:
        training_log = load_training_log(category)
        for key, value in training_log.items():
            if key == 'theta':
                # weights[category] = np.concatenate((value[-1][:6], [value[-1][7]]))
                weights[category] = value[-1][:]

     #Convert the weights into a pandas DataFrame
    weight_data = pd.DataFrame(weights, index=features)

    # Print the table (or save it to a CSV)
    print("Weight Table for Different Driving Styles:")
    print(weight_data)

    # Optionally, save the table to a CSV file
    weight_data.to_csv('weights_comparison_table.csv')


    # plot bar chart
    x = np.arange(len(features))  # the label locations
    width = 0.35  # the width of the bars
 
    # Create bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 1.5 * width, weights['Normal'], width, label='Normal', color='#E67E22',alpha=0.7)
    rects2 = ax.bar(x - 0.5 * width, weights['Aggressive'], width, label='Aggressive', color='#FF6F61',alpha=0.7)
    rects3 = ax.bar(x + 0.5 * width, weights['Cautious'], width, label='Cautious', color='#66BB6A',alpha=0.7)
    #rects4 = ax.bar(x + 1.5 * width, weights['All'], width, label='All', color='purple')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Features')
    ax.set_ylabel('Weights')
    ax.set_title('Weights for different driving styles')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=0)
    ax.legend()
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='gray')
    # Show the plot
    plt.tight_layout()
    #plt.savefig('weights_comparison.pdf', format='pdf')
    plt.show()


def main():
    categories = ['Normal', 'Aggressive', 'Cautious']
    plot_weight_comparison(categories)
    plot_training_process(categories[0])


if __name__ == "__main__":
    main()
