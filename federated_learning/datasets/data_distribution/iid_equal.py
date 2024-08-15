import torch
import numpy as np

# def distribute_batches_equally(train_data_loader, num_workers):
#     """
#     Gives each worker the same number of batches of training data.

#     :param train_data_loader: Training data loader
#     :type train_data_loader: torch.utils.data.DataLoader
#     :param num_workers: number of workers
#     :type num_workers: int
#     """
#     distributed_dataset = [[] for i in range(num_workers)]

#     for batch_idx, (data, target) in enumerate(train_data_loader):
#         worker_idx = batch_idx % num_workers

#         distributed_dataset[worker_idx].append((data, target))

#     return distributed_dataset

# ################### NON- IID ##################################
def distribute_batches_equally(train_data_loader, num_workers, alpha=0.5, K=10):
    """
    Distributes batches to workers in a non-IID manner using Dirichlet distribution.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: Number of workers
    :type num_workers: int
    :param alpha: Dirichlet distribution parameter, larger alpha means more balanced distribution
    :type alpha: float
    :param K: Number of classes in the dataset
    :type K: int
    :return: Distributed dataset and class counts for each worker
    :rtype: tuple (list, dict)
    """
    # Assuming train_data_loader is already defined
    X_train_list = []
    y_train_list = []
    print("NON-IID Alpha#{}",alpha)
    # print(alpha)
    for batch_idx, (data, target) in enumerate(train_data_loader):
        X_train_list.append(data)
        y_train_list.append(target)

    # Concatenate all batches to form the complete training set
    X_train = torch.cat(X_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)
    #print(len(X_train))
    #print(len(y_train))
    
    n_train = X_train.shape[0]
    min_size = 0
    N = y_train.shape[0]
    net_dataidx_map = {}
    

    while min_size < 10:
        idx_batch = [[] for _ in range(num_workers)]
        for k in range(K):
            idx_k = np.where(y_train.numpy() == k)[0]
            # print(idx_k)
            np.random.seed(42)
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_workers))
            proportions = np.array([p * (len(idx_j) < N / num_workers) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_workers):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    # # Create the distributed dataset
    distributed_dataset = [[] for _ in range(num_workers)]
    for worker_idx, data_indices in net_dataidx_map.items():
        worker_data = X_train[data_indices]
        worker_targets = y_train[data_indices]
        batch_size = train_data_loader.batch_size
        for i in range(0, len(worker_data), batch_size):
            batch_data = worker_data[i:i + batch_size]
            batch_target = worker_targets[i:i + batch_size]
            distributed_dataset[worker_idx].append((batch_data, batch_target))

    return distributed_dataset
