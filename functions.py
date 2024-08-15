from asyncio.log import logger
from torch.utils.data import TensorDataset, DataLoader
import random
import csv
import statistics
import numpy
import json
import pickle
import torch
torch.manual_seed(42)

#------------------------------------------#
#-----Random Worker Selection Strategy-----#
#------------------------------------------#

# def ReferenceData():
#     path = 'D:\Abhiram\amal_fl\1.FL\Attack\data_loaders\e-mnist\high_accuracy_test_data_loader_batched.pickle'

#     with open(path, 'rb') as f:
#         dataloader_batch = pickle.load(f)

#     #converting the list of tensors into tensors 
#     images1 = torch.cat([t[0] for t in dataloader_batch], dim=0)
#     labels1 = torch.cat([t[1] for t in dataloader_batch], dim=0)
#     # combined_tensor = torch.cat([images1, labels1], dim=0)

#     # print(images1.shape)
#     # print(labels1.shape)

#     #converting it to a tensor dataset and later to a dataloader of batch size 1000
#     dataset = TensorDataset(images1, labels1)
#     ref_dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)


#     return ref_dataloader




def round_worker(round,client,round_client,source,target,percentage,exp):
    key=(sum(source)*89)+(sum(target)*83)+(percentage*79)+(exp*89)
    random.seed(key)
    round_key=random.sample(list(range(1000)),round)
    round_workers=[]
    for i in round_key:
        random.seed(i)
        temp=random.sample(list(range(client)),round_client)
        temp.sort
        round_workers.append(temp)
    return round_workers
###poisoned worker selection
def poisioned_worker_selection(ROUND,KWARGS):
    return KWARGS["RANDOM_WORKERS"][ROUND][:KWARGS["NUM_POISONED_WORKERS"]]



#-----------------------------------------
#-----CSV genaration functions------------
#-----------------------------------------



def csv_gen(KWARGS,data):
    KWARGS_Copy=KWARGS.copy()
    del KWARGS_Copy["RANDOM_WORKERS"]
    with open('sheets/'+str(KWARGS["EXPID"])+'.csv', 'w') as file:
        writer = csv.writer(file)
        for row in KWARGS_Copy:
            writer.writerow([str(row),KWARGS_Copy[row]])
        writer.writerow([""])
        writer.writerow([""])
        writer.writerow([""])
        avgTestAcc=[]
        avgTrainAcc=[]
        avgSrcRcall=[]
        avgTargetRcall=[]
        avgSrcIndRcall=[]
        avgTargetIndRcall=[]
        avgRcallmal=[]
        avgRcallpure=[]
        avgSrcf1=[]
        avgTargetf1=[]
        avgSrcIndf1=[]
        avgTargetIndf1=[]
        avgf1mal=[]
        avgf1pure=[]
        AvgRcl =[]
        avg_ASR = []
        avg_miss_rate = []
        avg_tar_miss_rate = []
        writer.writerow(["Round","Random Workers","Test Avg Accuracy","Train Avg Accuracy","Source Avg recall","Targer Avg recall","Source individual recall","Target individual recall","Avg rcall including mal","Avg rcall Excluding mal","Source Avg F1score","Targer Avg F1score","Source individual F1score","Target individual F1score","Avg F1score including mal","Avg F1score Excluding mal","Rand Client Train Acc Before Update","All Client Train Acc After Update","All Client Test acc After Update","Updated model test score","server recalls","average of server recalls","Attack_success_rate","Misclassification_rate","Targeted_misclassification_rate"])
        for ra in range(KWARGS["ROUNDS"]):
            round_data=data[ra]
            workers=KWARGS["RANDOM_WORKERS"][ra]
            avgTestAcc.append(round(statistics.fmean(round_data[2]),3))
            avgTrainAcc.append(round(statistics.fmean(round_data[1]),3))
            avgSrcIndRcall.append(axis_avg(round_data[3],KWARGS["LABELS_TO_REPLACE"]))
            avgTargetIndRcall.append(axis_avg(round_data[3],KWARGS["LABELS_TO_REPLACE_WITH"]))
            avgSrcRcall.append(round(statistics.fmean(avgSrcIndRcall[ra]),3))
            avgTargetRcall.append(round(statistics.fmean(avgTargetIndRcall[ra]),3))
            avgRcallmal.append(round(recall_avg(round_data[3],[]),3))
            avgRcallpure.append(round(recall_avg(round_data[3],KWARGS["LABELS_TO_REPLACE"]+KWARGS["LABELS_TO_REPLACE_WITH"]),3))
            avgSrcIndf1.append(axis_avg(round_data[4],KWARGS["LABELS_TO_REPLACE"]))
            avgTargetIndf1.append(axis_avg(round_data[4],KWARGS["LABELS_TO_REPLACE_WITH"]))
            avgSrcf1.append(round(statistics.fmean(avgSrcIndf1[ra]),3))
            avgTargetf1.append(round(statistics.fmean(avgTargetIndf1[ra]),3))
            avgf1mal.append(round(recall_avg(round_data[4],[]),3))
            avgf1pure.append(round(recall_avg(round_data[4],KWARGS["LABELS_TO_REPLACE"]+KWARGS["LABELS_TO_REPLACE_WITH"]),3))
            AvgRcl.append(round(numpy.mean(round_data[5][3]),3))
            avg_ASR.append(round(round_data[5][5],3))
            avg_miss_rate.append(round(round_data[5][6],3))
            avg_tar_miss_rate.append(round(round_data[5][7],3))
            writer.writerow([ra,workers,avgTestAcc[ra],avgTrainAcc[ra],avgSrcRcall[ra],avgTargetRcall[ra],avgSrcIndRcall[ra],avgTargetIndRcall[ra],avgRcallmal[ra],avgRcallpure[ra],avgSrcf1[ra],avgTargetf1[ra],avgSrcIndf1[ra],avgTargetIndf1[ra],avgf1mal[ra],avgf1pure[ra],round_data[0],round_data[1],round_data[2],round_data[5][0],round_data[5][3],AvgRcl,round_data[5][5],avg_miss_rate[ra],avg_tar_miss_rate[ra]])

        writer.writerow([["Average Test Accuracy",round(statistics.fmean(avgTestAcc),3)]])
        writer.writerow([["Average Train Accuracy",round(statistics.mean(avgTrainAcc),3)]])
        writer.writerow([["Average Src Rcall",round(statistics.fmean(avgSrcRcall),3)]])
        writer.writerow([["Average Tget Rcall",round(statistics.fmean(avgTargetRcall),3)]])
        writer.writerow([["Average Rcall with MAl",round(statistics.fmean(avgRcallmal),3)]])
        writer.writerow([["Average Rcall without mal",round(statistics.fmean(avgRcallpure),3)]])
        writer.writerow([["Average Src F1score",round(statistics.fmean(avgSrcf1),3)]])
        writer.writerow([["Average Tget F1score",round(statistics.fmean(avgTargetf1),3)]])
        writer.writerow([["Average F1score with MAl",round(statistics.fmean(avgf1mal),3)]])
        writer.writerow([["Average F1score without mal",round(statistics.fmean(avgf1pure),3)]])
        final_data={
            "Average_Test_Accuracy":round(statistics.fmean(avgTestAcc),3),
            "Average_Train_Accuracy":round(statistics.mean(avgTrainAcc),3),
            "Average_Src_Rcall":round(statistics.fmean(avgSrcRcall),3),
            "Average Tget Rcall":round(statistics.fmean(avgTargetRcall),3),
            "Average Rcall with MAl":round(statistics.fmean(avgRcallmal),3),
            "Average Rcall without mal":round(statistics.fmean(avgRcallpure),3),
            "Average Src F1score":round(statistics.fmean(avgSrcf1),3),
            "Average Tget F1score":round(statistics.fmean(avgTargetf1),3),
            "Average F1score with MAl":round(statistics.fmean(avgf1mal),3),
            "Average F1score without mal":round(statistics.fmean(avgf1pure),3),
            "Average Recall":round(statistics.fmean(AvgRcl),3),
            "ASR":round(statistics.fmean(avg_ASR),3),
            "msr":round(statistics.fmean(avg_miss_rate),3),
            "tmr":round(statistics.fmean(avg_tar_miss_rate),3),
        }
    with open("sheets/"+str(KWARGS["EXPID"])+"final_data.json", "w") as outfile:
        outfile.write(json.dumps(final_data,indent=4))
    

def axis_avg(arr,col):
    ar=numpy.array(arr)
    avg=[]
    if col==[]:
        return [0]
    for i in col:
        avg.append(round(statistics.fmean(ar[:,i])*100,3))
    return avg
def recall_avg(arr,mal):
    ar=numpy.array(arr)
    for i in mal:
        ar[:,i]=0
    row_avg=[]
    lenmal=len(set(mal))
    for row in ar:
        row_avg.append((sum(row)/(len(row)-lenmal))*100)
    return statistics.fmean(row_avg)
def acc_select_rand_client(acc,clients):
    sel_acc=[]
    for i in clients:
        sel_acc.append(acc[i])
    return sel_acc




############ JSON functions ############


def saveParams(KWARGS,round,client,data):
    # file=open("models/"+str(KWARGS["EXPID"])+"_"+str(round)+"_"+str(client)+".sav",'wb')
    # pickle.dump(data,file)
    return 0

def To_csv(KWARGS,epoch,client_idx,rows):
        
    name_of_file = 'models/'+str(KWARGS['EXPID'])+'_'+str(epoch)+'_'+str(client_idx)+'_activations.csv'
    rows = numpy.vstack(rows)
    with open(name_of_file, 'w') as f:
        numpy.savetxt(f, rows, fmt='%f', delimiter=',')

def saveExpData(KWARGS):
    # with open("models/"+str(KWARGS["EXPID"])+".json", "w") as outfile:
    #     outfile.write(json.dumps(KWARGS,indent=4))
    return 0
    
####### Consolidation Function ##########
def consolidation(KWARGS,exp_id,exp_type):
    with open(exp_type+"-"+str(KWARGS["NUM_OF_REPLACEMENT"])+"L-"+str(KWARGS["NUM_POISONED_WORKERS"])+"C-"+str(KWARGS["PERCENTAGE_OF_REPLACEMENT"])+"%"+".csv", 'w') as outfile:
        csv_file=csv.writer(outfile)
        csv_file.writerow(["Exp_No","Test_Accuracy","Train_Accuracy","Source_Recall","Target_Recall","Rcall_with_malicious","Rcall_without_malicious","Source_f1score","Target_f1score","F1score_with_malicious","F1score_without_malicious"])
        arr=[[] for i in range(14)]
        for i in range(KWARGS["NUM_EXP"]):
            with open('sheets/'+str(exp_id+i)+'final_data.json', 'r') as file:
                json_file=json.load(file)
                arr[0].append(json_file["Average_Test_Accuracy"])
                arr[1].append(json_file["Average_Train_Accuracy"])
                arr[2].append(json_file["Average_Src_Rcall"])
                arr[3].append(json_file["Average Tget Rcall"])
                arr[4].append(json_file["Average Rcall with MAl"])
                arr[5].append(json_file["Average Rcall without mal"])
                arr[6].append(json_file["Average Src F1score"])
                arr[7].append(json_file["Average Tget F1score"])
                arr[8].append(json_file["Average F1score with MAl"])
                arr[9].append(json_file["Average F1score without mal"])
                arr[10].append(json_file["Average Recall"])
                arr[11].append(json_file["ASR"])
                arr[12].append(json_file["msr"])
                arr[13].append(json_file["tmr"])
                csv_file.writerow([i+1]+[li[i] for li in arr])
        csv_file.writerow(["AVG"]+[statistics.mean(li) for li in arr])

def databatcher():
    with torch.no_grad():
        data=[]
        with open('data_loaders/e-mnist/high_accuracy_data_loader_test.pickle','rb') as f:
            j=0
            for i in pickle.load(f):
                if j==0:
                    label=[]
                    image=[]
                    image.append(i[0])
                    label.append(i[1])
                else:
                    image.append(i[0])
                    label.append(i[1])
                j+=1
                if j==100:
                    j=0
                    image_tuple=tuple(image)
                    image_tensor=torch.cat(image_tuple)
                    label_tuple=tuple(label)
                    label_tensor=torch.cat(label_tuple)
                    data.append((image_tensor,label_tensor))
                print("Test data batched")
        with open('data_loaders/e-mnist/high_accuracy_test_data_loader_batched.pickle','wb') as f:
            pickle.dump(data,f)
        data=[]
        with open('data_loaders/e-mnist/high_accuracy_data_loader_train.pickle','rb') as f:
            j=0
            for i in pickle.load(f):
                if j==0:
                    label=[]
                    image=[]
                    image.append(i[0])
                    label.append(i[1])
                else:
                    image.append(i[0])
                    label.append(i[1])
                j+=1
                if j==1000:
                    j=0
                    image_tuple=tuple(image)
                    image_tensor=torch.cat(image_tuple)
                    label_tuple=tuple(label)
                    label_tensor=torch.cat(label_tuple)
                    data.append((image_tensor,label_tensor))
                print("Train Data Batched")
        with open('data_loaders/e-mnist/high_accuracy_train_data_loader_batched.pickle','wb') as f:
            pickle.dump(data,f)