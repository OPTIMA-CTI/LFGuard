from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.utils import average_nn_parameters
from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import identify_random_elements
from federated_learning.utils import save_results
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
from client import Client
from functions import saveParams,csv_gen,poisioned_worker_selection,saveExpData,To_csv
import time
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import csv
torch.manual_seed(42)
import joblib
#from defence_SVM import model
def train_subset_of_clients(epoch, args, clients,KWARGS,distributed_train_dataset):
    """
    Train a subset of clients per round.

    :param epoch: epoch
    :type epoch: int
    :param args: arguments
    :type args: Arguments
    :param clients: clients
    :type clients: list(Client)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    """
    # commented
    # kwargs = args.get_round_worker_selection_strategy_kwargs()
    # kwargs["current_epoch_number"] = epoch

    # random_workers = args.get_round_worker_selection_strategy().select_round_workers(
    #     list(range(args.get_num_workers())),
    #     poisoned_workers,
    #     kwargs)

    ##getting random worker
    random_workers=KWARGS["RANDOM_WORKERS"][epoch-1]
    ##Choosing poisoned worker
    poisoned_workers=poisioned_worker_selection(epoch-1,KWARGS)
    ##Genarating dataset
    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)
    if KWARGS["TARGETED_ATTACK"]:
        if(epoch>=100 and epoch<=150):
            distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers,KWARGS)
        else:
            distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), [],KWARGS)
    else:
        distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers,KWARGS)
        
    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())

    ##Training the Random Clients
    trainAccuracyOfRand=[]
    clientScores=[]
    RecallOfClients = []
    misclassified_of_clients=[]


    for client_idx in random_workers:
        np.set_printoptions(threshold=np.inf)

        
        args.get_logger().info("Training ,round #{} on client #{}", str(epoch), str(clients[client_idx].get_client_index()))
        acc,_=clients[client_idx].train(epoch,train_data_loaders[client_idx])

        trainAccuracyOfRand.append(acc)
        saveParams(KWARGS,epoch,client_idx,clients[client_idx].get_nn_parameters())

        #amal
        #get the activations of each model 

        activations, targets, preds = clients[client_idx].get_activation()
        flattened_activations = [a.flatten() for a in activations]
        df = pd.DataFrame(flattened_activations)
        #print(df.describe()) #3136 row
        
        df['targets'] = targets #original label of the image
        df['predictions'] = preds
        abc = list(df.columns)
        cols_to_round = abc[:-2]
        df[cols_to_round] = df[cols_to_round].apply(lambda x: x.round(4))
        df.columns = df.columns.astype(str)
        name_of_file = 'models/'+str(KWARGS['EXPID'])+'round_'+str(epoch)+'clientidx_'+str(client_idx)+'_activations.par'
        df.to_parquet(name_of_file)
        
        
        #print('defence running')
        #X_local = df.iloc[:,:2048]
        #y_local = df.iloc[:,2048]
        #predicted = model.predict(X_local)
        #print(model)
        #y_local=y_local.apply(lambda y: int(y[0]))


        #print('round',epoch,'client',client_idx)
        #y_local_arr = np.array(y_local)
        #cm = confusion_matrix(y_local_arr,predicted,labels=y_local.unique()) 
        #misclassified = np.sum(cm) - np.trace(cm)
        #print(misclassified)
        #misclassified_of_clients.extend([client_idx,misclassified])
        #clientScores.append(misclassified)
        
    #name_of_file = 'misclassification/'+str(KWARGS['EXPID'])+'round_'+str(epoch)+'.csv'
    # with open(name_of_file, 'w') as file:
    # # iterate over the list and write each item to a new line in the text file
    #     for item in misclassified_of_clients:
    #         file.write("%s\n" % item)
    # with open(name_of_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    
    # # Write the header row
    #     writer.writerow(['Index', 'missclassifications'])
    
    # # Iterate over the list and write each item to a new row in the CSV file
    #     for item in misclassified_of_clients:
    #         writer.writerow([item[0], item[1]])
    #ii = misclassified_of_clients[::2]
    #ss = misclassified_of_clients[1::2]
    
    
    #with open(name_of_file, mode='w', newline='') as file:
        #writer = csv.writer(file)
    
    # Write the header row
        #writer.writerow(['Index', 'Score'])
    
    # Iterate over the indices and scores and write them to the CSV file
        #for i in range(len(ii)):
            #writer.writerow([ii[i], ss[i]])


    #worst_client_indices = np.argsort(clientScores)[-3:] # indices to remove
    #print(worst_client_indices)
    #indices_of_attackers = [random_workers[i] for i in worst_client_indices]
    #args.get_logger().info("the clients to be removed are",indices_of_attackers)
    #new_clients = [client for idx, client in enumerate(random_workers) if idx not in worst_client_indices]
    #clients_for_aggregation = new_clients
    #args.get_logger().info("the clients considered for aggregation",clients_for_aggregation)
    
     # Print the list of clients for aggregation for this round
    #print(f"Round {str(epoch)}: Clients for aggregation: {clients_for_aggregation}")
    #clientScores.clear()
    args.get_logger().info("Averaging client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]
    

    new_nn_params = average_nn_parameters(parameters)
    saveParams(KWARGS,epoch,"avg",new_nn_params)
    
    #print(parameters)
    # _,activation=clients[client_idx].
    # print(activation)
    # activations=clients[client_idx].get_activation(train_data_loaders[client_idx])
    # print(activations)
    '''
    working

    activations=[clients[client_idx].get_activation(torch.randn(1,1,28,28)) for client_idx in random_workers]
    #print(activations)
    #print("shape=",activations[0][0].shape)
    '''
    '''
    working
    activations=clients[client_idx].get_activation(torch.randn(1,1,28,28))
    print(activations)
    '''
    # activations=[clients[client_idx].get_activation() for client_idx in random_workers]
    
    
    
    ## Updating and Training the All Clients
    trainAccuracyOfAll=[]
    testAccuracyOfAll=[]
    testRecallOfAll=[]
    testf1ofAll=[]
    client_idx=0

    for client in clients:
        args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
        client.update_nn_parameters(new_nn_params)
        if client_idx==0:
            server_test=client.server_test1(KWARGS["LABELS_TO_REPLACE"],KWARGS["LABELS_TO_REPLACE_WITH"])
        acc_train,_=client.train(epoch,train_data_loaders[client_idx])
        acc_test,_,_,recall_test,f1_test=client.test()
        trainAccuracyOfAll.append(acc_train)
        testAccuracyOfAll.append(acc_test)
        testRecallOfAll.append(recall_test)
        testf1ofAll.append(f1_test)
        client_idx+=1
    return [trainAccuracyOfRand,trainAccuracyOfAll,testAccuracyOfAll,testRecallOfAll,testf1ofAll,server_test]

def create_clients(args, test_data_loader):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, test_data_loader))

    return clients

def run_machine_learning(clients, args,KWARGS,distributed_train_dataset):
    """
    Complete machine learning over a series of clients.
    """
    #data for saving csv
    overall_data=[]
    for epoch in range(1, args.get_num_epochs() + 1):
        start_time=time.time()
        round_data= train_subset_of_clients(epoch, args, clients,KWARGS,distributed_train_dataset)
        overall_data.append(round_data)
        epoch_time=((time.time()-start_time)*(args.get_num_epochs()-epoch))
        logger.warning("Estimated time of Complition:{} minutes {} seconds",str(epoch_time//60),str(round(epoch_time%60,1)))
    csv_gen(KWARGS,overall_data)
def run_exp(KWARGS, client_selection_strategy, idx):
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)

    args = Arguments(logger)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(KWARGS["NUM_POISONED_WORKERS"])
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.log()
    args.set_num_epochs(KWARGS["ROUNDS"])
    args.set_num_workers(KWARGS["NUM_WORKERS"])

    
    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)
    KWARGS["EXPID"]=idx
    saveExpData(KWARGS)
    # Distribute batches equal volume IID
    distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())
    
    #print(distributed_train_dataset)
    
    # distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)
    # distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers)
    # train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())

    clients = create_clients(args, test_data_loader)

    run_machine_learning(clients, args,KWARGS,distributed_train_dataset)
    # save_results(results, results_files[0])
    # save_results(worker_selection, worker_selections_files[0])

    logger.remove(handler)
