from federated_learning.arguments import Arguments
from federated_learning.nets import Cifar10CNN
from federated_learning.nets import FashionMNISTCNN
from federated_learning.nets import EMNISTCNN
from federated_learning.nets import FED_EMNISTCNN
from federated_learning.nets import TrafficSignNet

import os
import torch
from loguru import logger

if __name__ == '__main__':
    args = Arguments(logger)
    if not os.path.exists(args.get_default_model_folder_path()):
        os.mkdir(args.get_default_model_folder_path())

    # # ---------------------------------
    # ----------- Cifar10CNN ----------
    # ---------------------------------
    #full_save_path = os.path.join(args.get_default_model_folder_path(), "Cifar10CNN.model")
    #torch.save(Cifar10CNN().state_dict(), full_save_path)

    # # ---------------------------------
    # # -------- FashionMNISTCNN --------
    # # ---------------------------------
    #full_save_path = os.path.join(args.get_default_model_folder_path(), "FashionMNISTCNN.model")
    #torch.save(FashionMNISTCNN().state_dict(), full_save_path)

    # ---------------------------------
    # -------- E-MNISTCNN --------
    # ---------------------------------
    # full_save_path = os.path.join(args.get_default_model_folder_path(), "EMNISTCNN.model")
    # torch.save(EMNISTCNN().state_dict(), full_save_path)

    # ---------------------------------
    # -------- FED-MNISTCNN --------
    # ---------------------------------
    # full_save_path = os.path.join(args.get_default_model_folder_path(), "FED_EMNISTCNN.model")
    # torch.save(FED_EMNISTCNN().state_dict(), full_save_path)
    # ---------------------------------
    # -------- GTSR --------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "TrafficSignNet.model")
    torch.save(TrafficSignNet().state_dict(), full_save_path)
