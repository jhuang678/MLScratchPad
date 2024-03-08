import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(42)

try:
    # Sample dataset creation
    n_samples = 200
    n_features = 5

    # Generating random data
    X = np.random.rand(n_samples, n_features)
    noise = np.random.normal(0, 0.2, n_samples)
    y = (
        2 * X[:, 0] +
        np.power(X[:, 1], 2) * 5 -
        np.cos(X[:, 2] * np.pi) * 3 +
        np.log(X[:, 3] + 1) * 4 +
        noise
    )

    # Creating a DataFrame
    df = pd.DataFrame(X, columns=[f'x_{i}' for i in range(n_features)])
    df['y'] = y

    # Saving the DataFrame to a CSV file
    df.to_csv('sample_data.csv', index=False)
    print("Generated successfully")

except Exception as e:
    print("Failed to generate")
    print("Error:", e)