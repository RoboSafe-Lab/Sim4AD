import numpy as np
import matplotlib.pyplot as plt


if __name__=="__main__":
    alpha = np.linspace(0, 1, 100)
    beta_values = [0.1, 0.0001]

    for beta in beta_values:
        y = beta ** (1-alpha)
        plt.plot(alpha, y, label=f'beta={beta}')

    plt.xlabel("Percentage of the original trajectory at which vehicle i went off-road")
    plt.ylabel('Off-road score')
    plt.title('Influence of beta on the off-road score')
    plt.legend()

    # Save as a svg file
    plt.savefig('beta_plot.svg')
    plt.show()