#!/usr/bin/env python
# Partitioning the population by Local PCA algorithm


# This function is translated from the Matlab code in
# http://bimk.ahu.edu.cn/index.php?s=/Index/Software/index.html


import numpy as np


def LocalPCA(PopDec, M, K):
    N, D = np.shape(PopDec)  # Dimensions
    Model = [{'mean': PopDec[k],  # The mean of the model
              'PI': np.eye(D),  # The matrix PI
              'eVector': [],  # The eigenvectors
              'eValue': [],  # The eigenvalues
              'a': [],  # The lower bound of the projections
              'b': []} for k in range(K)]  # The upper bound of the projections

    ## Modeling
    for iteration in range(1, 50):
        # Calculate the distance between each solution and its projection in
        # affine principal subspace of each cluster
        distance = np.zeros((N, K))  # matrix of zeros N*K
        for k in range(K):
            distance[:, k] = np.sum((PopDec - np.tile(Model[k]['mean'], (N, 1))).dot(Model[k]['PI']) * (
                    PopDec - np.tile(Model[k]['mean'], (N, 1))), 1)
        # Partition
        partition = np.argmin(distance, 1)  # get the index of mins
        # Update the model of each cluster
        updated = np.zeros(K, dtype=bool)  # array of k false
        for k in range(K):
            oldMean = Model[k]['mean']
            current = partition == k
            if sum(current) < 2:
                if not any(current):
                    current = [np.random.randint(N)]
                Model[k]['mean'] = PopDec[current, :]
                Model[k]['PI'] = np.eye(D)
                Model[k]['eVector'] = []
                Model[k]['eValue'] = []
            else:
                # print("'''''''")
                # print(PopDec[current, :])
                Model[k]['mean'] = np.mean(PopDec[current, :], 0)
                # print("<<<<>>>>")
                # print((PopDec[current, :] - np.tile(Model[k]['mean'], (np.sum(current), 1))).T)
                cc = np.cov((PopDec[current, :] - np.tile(Model[k]['mean'], (np.sum(current), 1))).T)

                eValue, eVector = np.linalg.eig(cc)
                rank = np.argsort(-(eValue), axis=0)
                eValue = -np.sort(-(eValue), axis=0)
                Model[k]['eValue'] = np.real(eValue).copy()
                Model[k]['eVector'] = np.real(eVector[:, rank]).copy()
                Model[k]['PI'] = Model[k]['eVector'][:, (M - 1):].dot(
                    Model[k]['eVector'][:, (M - 1):].conj().transpose())

            updated[k] = not any(current) or np.sqrt(np.sum((oldMean - Model[k]['mean']) ** 2)) > 1e-5

        # Break if no change is made
        if not any(updated):
            break

    ## Calculate the smallest hyper-rectangle of each model
    for k in range(K):
        if len(Model[k]['eVector']) != 0:
            hyperRectangle = (PopDec[partition == k, :] - np.tile(Model[k]['mean'], (sum(partition == k), 1))). \
                dot(Model[k]['eVector'][:, 0:M - 1])
            Model[k]['a'] = hyperRectangle.min(0)  # this should by tested
            Model[k]['b'] = hyperRectangle.max(0)  # this should by tested
        else:
            Model[k]['a'] = np.zeros((1, M - 1))
            Model[k]['b'] = np.zeros((1, M - 1))

    ## Calculate the probability of each cluster for reproduction
    # Calculate the volume of each cluster
    volume = []
    for k in range(K):
        try:
            b_val = np.atleast_1d(Model[k]['b']).flatten()
            a_val = np.atleast_1d(Model[k]['a']).flatten()
            
            # Ensure both arrays have the same length
            min_len = min(len(b_val), len(a_val))
            b_val = b_val[:min_len]
            a_val = a_val[:min_len]
            
            vol_diff = np.sum(b_val - a_val)  # Sum to get scalar value
            volume.append(vol_diff)
        except Exception as e:
            print(f"Warning: Error processing model {k}: {e}")
            volume.append(0.0)  # Default value

    volume = np.array(volume)  # Now should be scalar values, safe to convert

    # Calculate the cumulative probability of each cluster
    probability = np.cumsum(volume / np.sum(volume))
    return Model, probability