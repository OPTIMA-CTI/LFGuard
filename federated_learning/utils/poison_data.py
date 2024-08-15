# from .label_replacement import apply_class_label_replacement
from .client_utils import log_client_data_statistics
import random
###label flipping
def replace(target,x,y,z):
    #target: dataset label
    #x: data to flip
    #y: data to replace x
    #z: percentage of flip
    idx_list=[]
    for i in range(len(target)):
        if target[i]==x:
            idx_list.append(i)
    rand_selection=random.sample(idx_list,int(len(idx_list)*(z/100)))
    for i in rand_selection:
        target[i]=y
    return target
def apply_class_label_replacement(X, Y, KWARGS):
    """
    Replace class labels using the replacement method

    :param X: data features
    :type X: numpy.Array()
    :param Y: data labels
    :type Y: numpy.Array()
    :param replacement_data: data to apply the replacenet of label in dataset
    :type replacement_method: object
    """
    flip_count=KWARGS["NUM_OF_REPLACEMENT"]
    replace_label=KWARGS["LABELS_TO_REPLACE"]
    replacer_label=KWARGS["LABELS_TO_REPLACE_WITH"]
    replacement_percentage=KWARGS["PERCENTAGE_OF_REPLACEMENT"]
    labels=[]
    for i in range(flip_count):
        if i==0:
            labels=replace(Y,replace_label[i],replacer_label[i],replacement_percentage)
        else:
            labels=replace(labels,replace_label[i],replacer_label[i],replacement_percentage)
    return (X,labels)
def poison_data(logger, distributed_dataset, num_workers, poisoned_worker_ids, KWARGS):
    """
    Poison worker data

    :param logger: logger
    :type logger: loguru.logger
    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    :param num_workers: Number of workers overall
    :type num_workers: int
    :param poisoned_worker_ids: IDs poisoned workers
    :type poisoned_worker_ids: list(int)
    :param replacement_method: Replacement methods to use to replace
    :type replacement_method: list(method)
    """
    # TODO: Add support for multiple replacement methods?
    poisoned_dataset = []

    # class_labels = list(set(distributed_dataset[0][1]))
    label_class_set = set()
    for data, labels in distributed_dataset:
        label_class_set.update(labels)
    class_labels=list(label_class_set)

    logger.info("Poisoning data for workers: {}".format(str(poisoned_worker_ids)))

    for worker_idx in range(num_workers):
        if worker_idx in poisoned_worker_ids:
            poisoned_dataset.append(apply_class_label_replacement(distributed_dataset[worker_idx][0], distributed_dataset[worker_idx][1],KWARGS))
            # logger.info(distributed_dataset[worker_idx][1])
        else:
            poisoned_dataset.append(distributed_dataset[worker_idx])

    log_client_data_statistics(logger, class_labels, poisoned_dataset)

    return poisoned_dataset
