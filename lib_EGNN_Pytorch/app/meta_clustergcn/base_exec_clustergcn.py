### inside train module
import numpy as np
import time
import os

import torch

from ... import evaluation
from ...meta_learn import batch_machine
from ...models.model_app import GraphSAINT
from ... import utils



def prepare(working_dir, train_data, train_params, arch_gcn, num_clusters = 16, batch_num = 8):
    """
        working_dir: main working dir for experiments
        train_params: contain settings for the mini-batch setting
        arch_gcn: contain all the settings 
    """
    adj_full, adj_train, feat_full, class_arr, role = train_data
    adj_full = adj_full.astype(np.int32)  # change the original np.bool into 0-1 ints
    adj_train = adj_train.astype(np.int32)
    
    num_classes = class_arr.shape[1]
    
    # key switch :  cpu_eval (bool)
    # establish two models, one for train, one for evaluation, because later the model_eval will load the trained model parameters
    
#     # for training process: on GPU
    minibatch = batch_machine.Minibatch_clustergcn(adj_full, adj_train, role, train_params, cpu_eval = False, mode = "train", 
        num_clusters = num_clusters, batch_num = batch_num)

    model = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr)
    
    
    # for evaluation: validaiton/test  : on CPU
    minibatch_eval = batch_machine.Minibatch_clustergcn(adj_full, adj_train,role, train_params, cpu_eval=True, mode = "eval")
    model_eval = GraphSAINT(num_classes, arch_gcn, train_params, feat_full, class_arr, cpu_eval=True)
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    # model, model_eval, mini_batch_eval can be saved as pickle file for use later

    ### but cannot pickle lambda func for now
    prepare_data_folder = os.path.join(working_dir, 'prepare_data/')
    train_input_file_name = os.path.join(prepare_data_folder, 'model_train_input.pkl')
    utils.save_info_use_dill(train_input_file_name, (minibatch, model), exist_ok = True)
    
    evaluation_input_file_name = os.path.join(prepare_data_folder, 'model_eval_input.pkl')
    utils.save_info_use_dill(evaluation_input_file_name, (minibatch_eval, model_eval), exist_ok = True)
    

def train_investigate(snap_model_folder, train_phases, model, minibatch, eval_train_every, snapshot_every = 10,
          mini_epoch_num = 5, multilabel = True, core_par_sampler = 1, samples_per_processor = 200):
    """
    PURPOSE:  to go through each training phase and take a snapshot of current mode and saved as pickle files
        snap_model_folder : folder to save the model snapshots during training
        train_phases:  use defined train fases defined in the .yml file
        model :  graphsaint model for training
        minibatch:   minibatch for training, usually with batches pool
        eval_train_every :  periodically store the train loss during the training process
        snapshot_every :  periodically store the states of the trained model for later evaluation
        mini_epoch_num :  how long the training will focus on one single batch
        multilabel : True if a multi-label task, otherwise a multi-class case
        core_par_sampler : how many CPU cores  will be used on each CPU
        samples_per_processor : how many samples will be generated from each CPU
    return: 1) total training time;  2) data uploading time
    
    """
    
    epoch_ph_start = 0
    time_train, time_upload, pure_time_train = 0, 0, 0
    
    # establish a new folder if it does not exist
    
#     os.makedirs(snap_model_folder, exist_ok = True)
    num_batches = minibatch.num_training_batches
    for ip, phase in enumerate(train_phases):
        utils.printf('START PHASE {:4d}'.format(ip),style='underline')
        
#         print('calculated batch number is: ', num_batches)
        
        
        epoch_ph_end = int(phase['end'])
        macro_epoch_part_num = (epoch_ph_end - epoch_ph_start) // mini_epoch_num
        for macro_epoch_idx in range(macro_epoch_part_num):
            
            l_loss_tr, l_f1mic_tr, l_f1mac_tr = [], [], []
            
            actual_batch_num = 0
            
#             while not minibatch.end():
            # set the generator for all the one_batch
            train_batch_generator = minibatch.generate_train_batch(diag_lambda=-1)
            batch_idx = 0
            # iterate through the train batch generator:
            for node_subgraph, adj_subgraph, norm_loss_subgraph in train_batch_generator:
                
                ### =========== prepare for all the data to be used ==============
                feat_subg = model.feat_full[node_subgraph]
                label_subg = model.label_full[node_subgraph]
                
                if not multilabel:
                    # for the multi-class, need type conversion
                    label_full_cat = torch.from_numpy(model.label_full.numpy().argmax(axis=1).astype(np.int64))

                label_subg_converted = label_subg if multilabel else label_full_cat[node_subgraph]
                
                t0 = time.time()
                # transfer data to the GPU
                feat_subg = feat_subg.cuda()
                label_subg = label_subg.cuda()
                adj_subgraph = adj_subgraph.cuda()
                norm_loss_subgraph = norm_loss_subgraph.cuda()
                label_subg_converted = label_subg_converted.cuda()
                time_upload += time.time() - t0
                
                
                for micro_epoch_idx in range(mini_epoch_num):
                    real_epoch_idx = 1 + micro_epoch_idx + macro_epoch_idx * mini_epoch_num + epoch_ph_start
                    utils.printf('Epoch {:4d}, Batch ID {}'.format(real_epoch_idx, batch_idx),style='bold')
                    # pure training process:
                    t1 = time.time()
                    loss_train, preds_train = \
                            model.train_step(node_subgraph, adj_subgraph, norm_loss_subgraph, feat_subg, label_subg_converted)
                    
                    pure_time_train += time.time() - t1
                    labels_train = label_subg
                    
                    # take a snapshot of current model for later validation
                    if batch_idx == num_batches - 1 and real_epoch_idx % snapshot_every == 0:
                        snap_model_file = snap_model_folder + 'snapshot_epoch_' + str(real_epoch_idx) + '.pkl'
                        torch.save(model.state_dict(), snap_model_file)  # store the current state_dict() into the file: 'tmp.pkl'
                    
                    
                        # periodically calculate all the statistics and store them
#                             if not minibatch.batch_num % eval_train_every:
                        # utils.to_numpy already convert tensor onto CPU
                        f1_mic, f1_mac = evaluation.calc_f1(utils.to_numpy(labels_train), utils.to_numpy(preds_train), model.sigmoid_loss)
                        l_loss_tr.append(loss_train)
                        l_f1mic_tr.append(f1_mic)
                        l_f1mac_tr.append(f1_mac)
                
                # increase the iteration index of the train batch generator 
                batch_idx += 1
            
            
        # for different training phase, train it continuously 
        epoch_ph_start = int(phase['end'])
        utils.printf("Optimization Finished!", style="yellow")
    
    time_train = pure_time_train + time_upload
    # after going through all the training phases, print out the total time
    utils.printf("Total training time: {:6.2f} ms".format(time_train * 1000), style='red')
    utils.printf("Total train data uploading time: {:6.2f} ms".format(time_upload * 1000), style='red')
    
    return time_train * 1000, time_upload * 1000
            

