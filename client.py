import torch
import torch.optim as optim
from loguru import logger
from federated_learning.arguments import Arguments
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.schedulers import MinCapableStepLR
import torch.nn.functional as F
#from client import Client
# from federated_learning.nets import FED_EMNISTCNN
import os
import numpy
import copy
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy as np
path = 'data_loaders//gts//high_accuracy_test_data_loader_batched.pickle'
seed=42
torch.manual_seed(42)
with open(path, 'rb') as f:
    dataloader_batch = pickle.load(f)
#print(type(dataloader_batch)) #list
#print(len(dataloader_batch[0].shape))
#print(len(dataloader_batch[0][0]))
#batch_size = dataloader_batch.batch_size

#print(f"The batch size of the dataloader is: {batch_size}")


class Client:

    def __init__(self, args, client_idx,test_data_loader):
        """
        :param args: experiment arguments
        :type args: Arguments
        :param client_idx: Client index
        :type client_idx: int
        :param train_data_loader: Training data loader
        :type train_data_loader: torch.utils.data.DataLoader
        :param test_data_loader: Test data loader
        :type test_data_loader: torch.utils.data.DataLoader
        """
        self.args = args
        self.client_idx = client_idx

        self.device = self.initialize_device()
        self.set_net(self.load_default_model())

        self.loss_function = self.args.get_loss_function()()
        self.optimizer = optim.SGD(self.net.parameters(),
            lr=self.args.get_learning_rate(),
            momentum=self.args.get_momentum())
        self.scheduler = MinCapableStepLR(self.args.get_logger(), self.optimizer,
            self.args.get_scheduler_step_size(),
            self.args.get_scheduler_gamma(),
            self.args.get_min_lr())

        # self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

    def get_activation(self):
        


        #converting it to a tensor dataset and later to a dataloader of batch size 1000
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images1 = torch.cat([t[0] for t in dataloader_batch], dim=0)
        labels1 = torch.cat([t[1] for t in dataloader_batch], dim=0)

            
        dataset = TensorDataset(images1, labels1)
        ref_dataloader =DataLoader(dataset, batch_size=32, shuffle=True,worker_init_fn=np.random.seed(seed))
        iterator = iter(ref_dataloader)
        first_batch = tuple(t.to(device) for t in next(iterator))
        first_batch_dataset = TensorDataset(*first_batch)
        first_batch_dataloader = DataLoader(first_batch_dataset) #added
        

        self.net.eval()  
        args = Arguments(logger)
        
        with torch.no_grad():
            # correct = 0
            # total = 0
            activations = []
            targets = []
            preds = []
            for i, (inputs, labels) in enumerate(first_batch_dataloader, 0):
                
            
                inputs, labels = inputs.to(self.device), labels.to(self.device)
       
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                preds.append(predicted.cpu().numpy())
        
       
                if torch.cuda.is_available():
                # get the name of the current device
                    device = torch.cuda.get_device_name(torch.cuda.current_device())
                    #print("Using device:", device)
                else:
                    notex=1
                    #print("Cuda Is not Avalanle")
                # Extract activations of the last CNN layer
                # a = self.net.layer2(self.net.layer1(inputs))    
                x = F.relu(self.net.conv1(inputs))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.net.conv2(x))
                a = F.max_pool2d(x, 2)
                activations.append(a.cpu().numpy())
                targets.append(labels.cpu().numpy())
                
                
        return activations,targets,preds

    def initialize_device(self):
        """
        Creates appropriate torch device for client operation.
        """
        if torch.cuda.is_available() and self.args.get_cuda():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")

    def set_net(self, net):
        """
        Set the client's NN.

        :param net: torch.nn
        """
        self.net = net
        self.net.to(self.device)

    def load_default_model(self):
        """
        Load a model from default model file.

        This is used to ensure consistent default model behavior.
        """
        model_class = self.args.get_net()
        default_model_path = os.path.join(self.args.get_default_model_folder_path(), model_class.__name__ + ".model")

        return self.load_model_from_file(default_model_path)

    def load_model_from_file(self, model_file_path):
        """
        Load a model from a file.

        :param model_file_path: string
        """
        model_class = self.args.get_net()
        model = model_class()

        if os.path.exists(model_file_path):
            try:
                model.load_state_dict(torch.load(model_file_path))
            except:
                self.args.get_logger().warning("Couldn't load model. Attempting to map CUDA tensors to CPU to solve error.")

                model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
        else:
            self.args.get_logger().warning("Could not find model: {}".format(model_file_path))

        return model

    def get_client_index(self):
        """
        Returns the client index.
        """
        return self.client_idx

    def get_nn_parameters(self):
        """
        Return the NN's parameters.
        """
        return self.net.state_dict()
    

    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        :type new_params: dict
        """
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)

    def train(self, epoch,traindata):
        """
        :param epoch: Current epoch #
        :type epoch: int
        """
        self.net.train()

        correct = 0
        total = 0
        l = 0
        targets_ = []
        pred_ = []

        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_start_suffix())

        running_loss = 0.0
       
        correct = 0
        total = 0
        l = 0
        targets_ = []
        pred_ = []
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(traindata, 0):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            targets_.extend(labels.cpu().view_as(predicted).numpy())
            pred_.extend(predicted.cpu().numpy())

            ## By me
            # print statistics
            running_loss += loss.item()
            l += loss.item()
            if i % self.args.get_log_interval() == 0:
                self.args.get_logger().info('[%d, %5d] loss: %.3f' % (epoch, i,running_loss / self.args.get_log_interval()))

                running_loss = 0.0

        accuracy = numpy.around(100 * correct / total , 3)
        l = numpy.around(l/total,2)

        self.args.get_logger().debug('Train set: Accuracy : {}/{} ({:.3f}%)'.format(correct, total, accuracy))
        self.args.get_logger().debug('Train set: Loss: ({:.3f})'.format(l))
        self.scheduler.step()

        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())

        return accuracy , l



    def save_model(self, epoch, suffix):
        """
        Saves the model if necessary.
        """
        self.args.get_logger().debug("Saving model to flat file storage. Save #{}", epoch)

        if not os.path.exists(self.args.get_save_model_folder_path()):
            os.mkdir(self.args.get_save_model_folder_path())

        full_save_path = os.path.join(self.args.get_save_model_folder_path(), "model_" + str(self.client_idx) + "_" + str(epoch) + "_" + suffix + ".model")
        torch.save(self.get_nn_parameters(), full_save_path)

    def calculate_class_precision(self, confusion_mat):
        """
        Calculates the precision for each class from a confusion matrix.
        """
        return numpy.diagonal(confusion_mat) / numpy.sum(confusion_mat, axis=0)

    def calculate_class_recall(self, confusion_mat):
        """
        Calculates the recall for each class from a confusion matrix.
        """
        return numpy.diagonal(confusion_mat) / numpy.sum(confusion_mat, axis=1)
    def calculate_class_f1score(self, confusion_mat):
        """
        Calculates the recall for each class from a confusion matrix.
        """
        return numpy.diagonal(confusion_mat) / numpy.sum(confusion_mat, axis=2)

    def test(self):
        self.net.eval()

        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        with torch.no_grad():
            for (images, labels) in self.test_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()

        accuracy = numpy.around(100 * correct / total , 3)
        confusion_mat = confusion_matrix(targets_, pred_)

        class_precision = self.calculate_class_precision(confusion_mat)
        class_recall = self.calculate_class_recall(confusion_mat)
        class_f1score=f1_score(targets_,pred_,average=None)

        self.args.get_logger().debug('Test set: Accuracy: {}/{} ({}%)'.format(correct, total, accuracy))
        self.args.get_logger().debug('Test set: Loss: {}'.format(loss))
        self.args.get_logger().debug("Classification Report:\n" + classification_report(targets_, pred_))
        self.args.get_logger().debug("Confusion Matrix:\n" + str(confusion_mat))
        self.args.get_logger().debug("Class precision: {}".format(str(class_precision)))
        self.args.get_logger().debug("Class recall: {}".format(str(class_recall)))
        self.args.get_logger().debug("Class f1score: {}".format(str(class_f1score)))

        return accuracy, loss, class_precision, class_recall,class_f1score
    
    def server_test1(self,actual_prediction,target_prediction):
        self.net.eval()
        n = len(actual_prediction)
        d = dict(zip(actual_prediction,target_prediction))
        print(d)
        actual_prediction = torch.Tensor(actual_prediction)
        target_prediction = torch.Tensor(target_prediction)
        correct = 0
        attack_success_count = 0
        instances = 0
        total = 0
        misclassifications = 0
        targeted_misclassification = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        with torch.no_grad():
            for (images, labels) in self.test_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()
                 
                #code added
                for i in range(len(labels)):
                    if len(list(d.keys())) ==0:
                        attack_success_count=0
                        misclassifications=0
                        targeted_misclassification=0
                        instances=1
                        total=1
                        n=1
                        
                        break
                    for j in list(d.keys()):
                        #print(d[j])
                        #j = torch.Tensor(j)
                        #print(j)
                        
                        #d[j] = torch.Tensor(d[j])
                        targ = [int(d[j])]
                        act = [int(j)]
                        
                        #print(j)
                        #print(torch.Tensor(j))
                        
                        if labels[i] == torch.Tensor(act).to(labels.device):
                            instances += 1
                        if labels[i] != predicted[i]:
                            misclassifications += 1
                            if labels[i] == j:
                                targeted_misclassification += 1
                                if predicted[i] == torch.Tensor(targ).to(labels.device):
                                    attack_success_count +=1
                            
                        
                        

        accuracy = numpy.around(100 * correct / total , 3)
        confusion_mat = confusion_matrix(targets_, pred_)

        class_precision = self.calculate_class_precision(confusion_mat)
        class_recall = self.calculate_class_recall(confusion_mat)
        class_f1score=f1_score(targets_,pred_,average=None)
        #code added
        
        total = total * n
        attack_success_rate = attack_success_count/instances*100
        misclassification_rate = misclassifications/total*100
        targeted_misclassification_rate = targeted_misclassification/instances *100

        self.args.get_logger().debug('Test set: Accuracy: {}/{} ({}%)'.format(correct, total, accuracy))
        self.args.get_logger().debug('Test set: Loss: {}'.format(loss))
        self.args.get_logger().debug("Classification Report:\n" + classification_report(targets_, pred_))
        self.args.get_logger().debug("Confusion Matrix:\n" + str(confusion_mat))
        self.args.get_logger().debug("Class precision: {}".format(str(class_precision)))
        self.args.get_logger().debug("Class recall: {}".format(str(class_recall)))
        self.args.get_logger().debug("Class f1score: {}".format(str(class_f1score)))
        self.args.get_logger().debug("Attack_success_rate: {}".format(str(attack_success_rate)))
        self.args.get_logger().debug("misclassification_rate: {}".format(str(misclassification_rate)))
        self.args.get_logger().debug("targeted_misclassification_rate: {}".format(str(targeted_misclassification_rate)))
        self.args.get_logger().debug("instances: {}".format(str(instances)))
        self.args.get_logger().debug("ASC: {}".format(str(attack_success_count)))
        self.args.get_logger().debug("missclassification: {}".format(str(misclassifications)))
        self.args.get_logger().debug("targeted_misclassification {}".format(str(targeted_misclassification)))
        self.args.get_logger().debug("Total: {}".format(str(total)))

        return accuracy, loss, class_precision, class_recall,class_f1score,attack_success_rate,misclassification_rate,targeted_misclassification_rate
