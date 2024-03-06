######################################################################
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#####################################################################

def calculate_epsilon(D_t, y_t, h_t):
    epsilon_t = np.sum(D_t * (y_t != h_t))
    return epsilon_t

def calculate_alpha(epsilon_t):
    return math.log(((1 - epsilon_t)/epsilon_t))/2

def calculate_Z(D_t, alpha_t, y_t, h_t):
    return np.sum(D_t * np.exp(-alpha_t * y_t * h_t))

def update_D(D_t, alpha_t, y_t, h_t, Z_t):
    return (D_t / Z_t) * np.exp(-alpha_t * y_t * h_t)

def misclassification_rate(Y, h):
    return np.sum(Y != h)/ len(Y)

if __name__ == '__main__':
    data = [(-1, 0, 1), (-0.5, 0.5, 1), (0, 1, -1), (0.5, 1, -1),
            (1, 0, 1), (1, -1, 1), (0, -1, -1), (0, 0, -1)]
    X = [(x1, x2) for x1, x2, _ in data]
    Y = np.array([label for _, _, label in data])

    labels = [f"X{i + 1}" for i in range(len(data))]  # Generate labels like "X_1", "X_2", ...
    X_pos = [X[i] for i in range(len(X)) if Y[i] == 1]
    X_neg = [X[i] for i in range(len(X)) if Y[i] == -1]
    b = 1.5

    print("t = 1")
    # Initialize D
    D1 = np.array([1 / 8] * 8)
    S_pos = tuple([400*D1[i] for i in range(len(D1)) if Y[i] == 1])
    S_neg = tuple([400*D1[i] for i in range(len(D1)) if Y[i] == -1])
    # Scatter plot
    plt.xlim(-b, b)
    plt.ylim(-b, b)
    plt.scatter(*zip(*X_pos), c='red', marker='+', label='Positive (+1)', s=S_pos, zorder=10)
    plt.scatter(*zip(*X_neg), marker='o', facecolors='none', edgecolors='blue', label='Negative (-1)', s=S_neg, zorder=10)
    for (x, y), label in zip(X, labels):
        plt.text(x + 0.05, y, label, fontsize=9, verticalalignment='center', zorder=9)
    # Base Classifier h1
    h1 = -0.25
    plt.axvline(x=h1, color='black', linestyle='--', linewidth=2, zorder=10)
    plt.axvspan(h1, b, facecolor='b', alpha=0.1)
    plt.axvspan(-b,h1, facecolor='r', alpha=0.1)
    # Complete chart
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('AdaBoost (t=1)')
    plt.grid(True, zorder=-10)
    plt.savefig('t1.png')
    plt.close()

    # Calculate Result
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
    mis_rate = misclassification_rate(Y, h1)
    e1 = calculate_epsilon(D_t=D1, y_t=Y, h_t=h1)
    a1 = calculate_alpha(epsilon_t=e1)
    Z1 = calculate_Z(D_t=D1, alpha_t=a1, y_t=Y, h_t=h1)

    print("e1: ", round(e1,4))
    print("a1: ", round(a1,4))
    print("Z1: ", round(Z1,4))
    print("D1: ", [round(d, 4) for d in D1])
    print("Misclassification Rate:", round(mis_rate,4))
    print()


    print("t = 2")
    # Calculate D
    D2 = update_D(D_t=D1, alpha_t=a1, y_t=Y, h_t=h1, Z_t=Z1)
    S_pos = tuple([400 * D2[i] for i in range(len(D2)) if Y[i] == 1])
    S_neg = tuple([400 * D2[i] for i in range(len(D2)) if Y[i] == -1])
    # Scatter plot
    plt.xlim(-b, b)
    plt.ylim(-b, b)
    plt.scatter(*zip(*X_pos), c='red', marker='+', label='Positive (+1)', s=S_pos, zorder=10)
    plt.scatter(*zip(*X_neg), marker='o', facecolors='none', edgecolors='blue', label='Negative (-1)', s=S_neg,
                zorder=10)
    for (x, y), label in zip(X, labels):
        plt.text(x + 0.05, y, label, fontsize=9, verticalalignment='center', zorder=9)
    # Base Classifier h1
    h2 = 0.75
    plt.axvline(x=h2, color='black', linestyle='--', linewidth=2, zorder=10)
    plt.axvspan(h2, b, facecolor='r', alpha=0.1)
    plt.axvspan(-b, h2, facecolor='b', alpha=0.1)
    # Complete chart
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('AdaBoost (t=2)')
    plt.grid(True, zorder=-10)
    plt.savefig('t2.png')
    plt.close()

    # Calculate Result
    h2 = np.array([-1, -1, -1, -1, 1, 1, -1, -1])
    mis_rate = misclassification_rate(Y, h2)
    e2 = calculate_epsilon(D_t=D2, y_t=Y, h_t=h2)
    a2 = calculate_alpha(epsilon_t=e2)
    Z2 = calculate_Z(D_t=D2, alpha_t=a2, y_t=Y, h_t=h2)

    print("e2: ", round(e2, 4))
    print("a2: ", round(a2, 4))
    print("Z2: ", round(Z2, 4))
    print("D2: ", [round(d, 4) for d in D2])
    print("Misclassification Rate:", round(mis_rate, 4))
    print()

    print("t = 3")
    # Calculate D
    D3 = update_D(D_t=D2, alpha_t=a2, y_t=Y, h_t=h2, Z_t=Z2)
    S_pos = tuple([400 * D3[i] for i in range(len(D3)) if Y[i] == 1])
    S_neg = tuple([400 * D3[i] for i in range(len(D3)) if Y[i] == -1])
    # Scatter plot
    plt.xlim(-b, b)
    plt.ylim(-b, b)
    plt.scatter(*zip(*X_pos), c='red', marker='+', label='Positive (+1)', s=S_pos, zorder=10)
    plt.scatter(*zip(*X_neg), marker='o', facecolors='none', edgecolors='blue', label='Negative (-1)', s=S_neg,
                zorder=10)
    for (x, y), label in zip(X, labels):
        plt.text(x + 0.05, y, label, fontsize=9, verticalalignment='center', zorder=9)
    # Base Classifier h1
    h3 = 0.75
    plt.axhline(y=h3, color='black', linestyle='--', linewidth=2, zorder=10)
    plt.axhspan(h3, b, facecolor='b', alpha=0.1)
    plt.axhspan(-b, h3, facecolor='r', alpha=0.1)
    # Complete chart
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('AdaBoost (t=3)')
    plt.grid(True, zorder=-10)
    plt.savefig('t3.png')
    plt.close()

    # Calculate Result
    h3 = np.array([1, 1, -1, -1, 1, 1, 1, 1])
    mis_rate = misclassification_rate(Y, h3)
    e3 = calculate_epsilon(D_t=D3, y_t=Y, h_t=h3)
    a3 = calculate_alpha(epsilon_t=e3)
    Z3 = calculate_Z(D_t=D3, alpha_t=a3, y_t=Y, h_t=h3)

    print("e3: ", round(e3, 4))
    print("a3: ", round(a3, 4))
    print("Z3: ", round(Z3, 4))
    print("D3: ", [round(d, 4) for d in D3])
    print("Misclassification Rate:", round(mis_rate, 4))
    print()

    print("Final AdaBoost Prediction Accuracy")
    H = np.sign(a1*h1 + a2*h2 + a3*h3)
    print(H)
    mis_rate = misclassification_rate(Y, H)
    print("Misclassification Rate:", round(mis_rate, 4))




