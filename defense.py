from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp
from functions import round_worker,consolidation
import os
from pathlib import Path
if __name__ == '__main__':
    START_EXP_IDX = 300 ####Always Start with oned Index eg:3001
    NUM_EXP = 1
    # NUM_POISONED_WORKERS=1
    NUM_WORKERS_PER_ROUND=10
    NUM_OF_REPLACEMENT=3
    LABELS_TO_REPLACE=[0,7,8]
    LABELS_TO_REPLACE_WITH=[4,9,6]
    PERCENTAGE_OF_REPLACEMENT=10
    ROUNDS=200
    NUM_WORKERS=20
    LOCAL_EPOCH=1
    for NUM_POISONED_WORKERS in range(3,4):

        for PERCENTAGE_OF_REPLACEMENT in range(50, 55, 10):
            for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
                RANDOM_WORKERS=round_worker(ROUNDS,NUM_WORKERS,NUM_WORKERS_PER_ROUND,[0],[9],10,experiment_id)  # 9 7
                KWARGS = {
                    "EXPID":0,
                    "ROUNDS":ROUNDS,
                    "NUM_WORKERS":NUM_WORKERS,
                    
                    "NUM_EXP":NUM_EXP,
                    "NUM_POISONED_WORKERS":NUM_POISONED_WORKERS,
                    "NUM_WORKERS_PER_ROUND" : NUM_WORKERS_PER_ROUND,
                    "RANDOM_WORKERS":RANDOM_WORKERS,
                    "NUM_OF_REPLACEMENT":NUM_OF_REPLACEMENT,
                    "LABELS_TO_REPLACE":LABELS_TO_REPLACE,
                    "LABELS_TO_REPLACE_WITH":LABELS_TO_REPLACE_WITH,
                    "PERCENTAGE_OF_REPLACEMENT":PERCENTAGE_OF_REPLACEMENT,
                    
                    "LOCAL_EPOCH":LOCAL_EPOCH,
                    
                }
                run_exp(KWARGS, RandomSelectionStrategy(), experiment_id)
                new_sheets_name = f"sheets_{NUM_OF_REPLACEMENT}_{NUM_POISONED_WORKERS}_{PERCENTAGE_OF_REPLACEMENT}"
                new_logs_name = f"logs_{NUM_OF_REPLACEMENT}_{NUM_POISONED_WORKERS}_{PERCENTAGE_OF_REPLACEMENT}"
                new_confusion_name = f"confusion_matrix_{NUM_OF_REPLACEMENT}_{NUM_POISONED_WORKERS}_{PERCENTAGE_OF_REPLACEMENT}"
                os.rename("sheets",new_sheets_name)
                os.rename("logs",new_logs_name)
                os.rename("confusion_matrix",new_confusion_name)
                # Specify the name of the new folder
                folder_name = "sheets"
                folder_name2="logs"
                folder_name3="confusion_matrix"

                # Get the current working directory
                cwd = Path.cwd()

                # Use / operator to join the current working directory with the new folder name
                folder_path = cwd/folder_name
                folder_path2=cwd/folder_name2
                folder_path3=cwd/folder_name3
                # Use Path.mkdir() to create the folder
                folder_path.mkdir()
                folder_path2.mkdir()
                folder_path3.mkdir()
