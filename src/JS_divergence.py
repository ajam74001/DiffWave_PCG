import numpy as np
import pandas as pd

root_dir = '/home/ainazj1/psanjay_ada/users/ainazj1/Datasets/physionet.org/files/circor-heart-sound/1.0.3'
real = pd.read_csv(root_dir+'/real_diffwave_final.csv').drop('Unnamed: 0', axis=1).dropna()
fake = pd.read_csv(root_dir+'/fake_diffwave_final.csv').drop('Unnamed: 0', axis=1).dropna() 


def timeseries_to_hist(data):
    df_data=[]
    for i in range(data.shape[0]):
        df_data.append(data.iloc[i, :])
    flattened_data = np.concatenate(df_data)
    hist, bin_edges = np.histogram(flattened_data, bins=557, density=True) # the value for the bin is suggested by the auto. I set it fixed cuz the auto was suggesting different values which leads to calcuation errors in the functions 
    # print(len(bin_edges))
    return hist

# print(real.iloc[ :len(fake),:].shape)
real_hist = timeseries_to_hist(real.iloc[ :len(fake),:])
print(real_hist.shape)
fake_hist = timeseries_to_hist(fake)

def jensen_shannon_divergence(p, q):
    """
    Inputs: two probabilty distribution 
    """
    p = np.asarray(p, dtype=np.float64)
    p /= np.sum(p)
    q = np.asarray(q, dtype=np.float64)
    q /= np.sum(q)

    # Calculate the average distribution
    avg = 0.5 * (p + q)

    # Add a small constant to avoid division by zero and log of zero
    epsilon = 1e-10
    p_safe = np.maximum(p, epsilon)
    q_safe = np.maximum(q, epsilon)
    avg_safe = np.maximum(avg, epsilon)

    # Calculate the JS divergence
    jsd = 0.5 * (np.sum(p_safe * np.log2(p_safe / avg_safe)) + np.sum(q_safe * np.log2(q_safe / avg_safe)))
    return jsd
 
print("jensen_shannon_divergence ", jensen_shannon_divergence(real_hist, fake_hist))





def maximum_mean_discrepancy(X, Y, kernel=np.exp):
    # Convert the arrays to numpy arrays if they're not already
    X = np.array(X)
    Y = np.array(Y)
    
    # Number of samples in X and Y
    n, m = X.shape[0], Y.shape[0]

    # Compute the kernel matrices
    XX = kernel(np.dot(X, X.T))
    YY = kernel(np.dot(Y, Y.T))
    XY = kernel(np.dot(X, Y.T))
    
    # Calculate the MMD^2 statistic
    mmd_squared = 1.0 / (n * (n - 1)) * np.sum(XX) - 2.0 / (n * m) * np.sum(XY) + 1.0 / (m * (m - 1)) * np.sum(YY)
    
    return np.sqrt(mmd_squared)

print( "maximum_mean_discrepancy ", maximum_mean_discrepancy(real_hist, fake_hist))

