import shutil
import os
import logging

from . import basic_exec_cluster, basic_exec_link, basic_exec_node_classification
from ... import utils


def execute_train_investigate(data_name, model, input_data, workdir, config, 
                              tune_param_name, tune_val_label, tune_val, trainer_id = 0, 
                              device = "cpu"):
    """
        Train wrapper for tuning hyper-parameters during training
            1) May include a list of tuning values
            2) for each training value, include repeated experiments, multi-trainer_id

        Args:
            data_file_path : Already pre-processed input dataset
            workdir : code execution main directory
            config : yaml file containing main settings
    """
    # >>>>>>>>>>>>>>>>>>>> establish a new pprgo model, set up the model checkpoint for best validation metrics
    checkpoint_file_path = os.path.join(workdir,
                                f"tune_{tune_param_name}/model_checkpoint/tunelabel_{tune_val_label}_trainer_{trainer_id}/best_model.pkl")

    if os.path.exists(checkpoint_file_path):
        print("ckpt file already exists, so removed ...")
        os.remove(checkpoint_file_path)
    else:
        os.makedirs(os.path.dirname(checkpoint_file_path), exist_ok=True)

    val_metric_path = os.path.join(workdir, 
                                f"tune_{tune_param_name}/val_metric/tunelabel_{tune_val_label}_trainer_{trainer_id}/val_metric.pkl")

    # ==========================  Start the training ==========================
    time_training, metric_summary = basic_exec_cluster.train(model, config, input_data, device = device, checkpoint_file_path = checkpoint_file_path)
    
    logging.info(f"Training task (tune_param_name: {tune_param_name}; tune_val: {tune_val}; trainer_id: {trainer_id}) take: {time_training:.3f} seconds")
    # ==========================  End of the training ==========================
    
    # <<<<<<<<<<<<<<<<<<<<<< Save the train loss and validation metrics
    
    utils.save_info_pickle(data_path = val_metric_path, target_data = metric_summary )
    
    # <<<<<<<<<<<<<<<<<<<<<< Save the profiling info during training
    
    train_profile_folder = os.path.join(workdir, f"tune_{tune_param_name}/train_profile/tunelabel_{tune_val_label}_trainer_{trainer_id}/")                       
    train_profile_path = os.path.join(train_profile_folder, "train_profile.pkl")
    
    utils.save_info_pickle(data_path = train_profile_path, target_data = {"time_training" : time_training})        
    

def execute_test_investigate(data_name, new_model, input_data, workdir, config, 
                             tune_param_name, tune_val_label, tune_val, trainer_id = 0):
    """
        Train wrapper for tuning hyper-parameters during training
            1) May include a list of tuning values
            2) for each training value, include repeated experiments, multi-trainer_id
    """
    # >>>>>>>>>>>>>>>>>>>> establish a new pprgo model, set up the model checkpoint for best validation metrics
    checkpoint_file_path = os.path.join(workdir,
                                f"tune_{tune_param_name}/model_checkpoint/tunelabel_{tune_val_label}_trainer_{trainer_id}/best_model.pkl")

    if not os.path.exists(os.path.dirname(checkpoint_file_path)):
        raise("checkpoint file is missing")
    
    test_metric_path = os.path.join(workdir,
                                f"tune_{tune_param_name}/test_metric/tunelabel_{tune_val_label}_trainer_{trainer_id}/test_metric.pkl")

    # >>>>>>>>>>>>>>>>>>>> Start test inference
    test_time, test_metric = basic_exec_cluster.test(new_model, config, input_data, 
                                device = "cpu", checkpoint_file_path = checkpoint_file_path)
    
    print(f"Test Runtime: {test_time:.2f}s")
    
    # <<<<<<<<<<<<<<<<<<<<<< Save the train loss and validation metrics
    
    
    utils.save_info_pickle(data_path = test_metric_path, target_data = test_metric)
    
    # <<<<<<<<<<<<<<<<<<<<<< Save the profiling info during training
    test_profile_path = os.path.join(workdir,
                                f"tune_{tune_param_name}/test_profile/tunelabel_{tune_val_label}_trainer_{trainer_id}/test_profile.pkl")
    
    utils.save_info_pickle(data_path = test_profile_path, 
                     target_data = {"test_time" : test_time})



#### ============================ For link prediction  ========================

def execute_train_link(data_name, model, input_data, workdir, config, 
                              tune_param_name, tune_val_label, tune_val, trainer_id = 0, device = "cpu"):
    """
        Train wrapper for tuning hyper-parameters during training
            1) May include a list of tuning values
            2) for each training value, include repeated experiments, multi-trainer_id

        Args:
            data_file_path : Already pre-processed input dataset
            workdir : code execution main directory
            config : yaml file containing main settings
    """
    # >>>>>>>>>>>>>>>>>>>> establish a new pprgo model, set up the model checkpoint for best validation metrics
    checkpoint_file_path = os.path.join(workdir,
                                f"tune_{tune_param_name}/model_checkpoint/tunelabel_{tune_val_label}_trainer_{trainer_id}/best_model.pkl")

    if os.path.exists(os.path.dirname(checkpoint_file_path)):
        print("ckpt folder already exists, so removed ...")
        shutil.rmtree(os.path.dirname(checkpoint_file_path))
    os.makedirs(os.path.dirname(checkpoint_file_path), exist_ok=True)

    # ==========================  Start the training ==========================
    val_metric_path = os.path.join(workdir, 
                                f"tune_{tune_param_name}/val_metric/tunelabel_{tune_val_label}_trainer_{trainer_id}/val_metric.pkl")

    time_training, metric_summary = basic_exec_link.train(model, config, input_data, device = device, checkpoint_file_path = checkpoint_file_path)
    
    logging.info(f"Training task (tune_param_name: {tune_param_name}; tune_val: {tune_val}; trainer_id: {trainer_id}) take: {time_training:.3f} seconds")
    # ==========================  End of the training ==========================
    
    # <<<<<<<<<<<<<<<<<<<<<< Save the train loss and validation metrics
    
    utils.save_info_pickle(data_path = val_metric_path, target_data = metric_summary )
    
    # <<<<<<<<<<<<<<<<<<<<<< Save the profiling info during training
    
    train_profile_folder = os.path.join(workdir, f"tune_{tune_param_name}/train_profile/tunelabel_{tune_val_label}_trainer_{trainer_id}/")                       
    train_profile_path = os.path.join(train_profile_folder, "train_profile.pkl")
    
    utils.save_info_pickle(data_path = train_profile_path, target_data = {"time_training" : time_training})  
    

def execute_test_link(data_name, new_model, input_data, workdir, config, 
                             tune_param_name, tune_val_label, tune_val, trainer_id = 0):
    """
        Train wrapper for tuning hyper-parameters during training
            1) May include a list of tuning values
            2) for each training value, include repeated experiments, multi-trainer_id
    """
    # >>>>>>>>>>>>>>>>>>>> establish a new pprgo model, set up the model checkpoint for best validation metrics
    checkpoint_file_path = os.path.join(workdir,
                                f"tune_{tune_param_name}/model_checkpoint/tunelabel_{tune_val_label}_trainer_{trainer_id}/best_model.pkl")

    if not os.path.exists(os.path.dirname(checkpoint_file_path)):
        raise("checkpoint file is missing")

    test_metric_path = os.path.join(workdir,
                                f"tune_{tune_param_name}/test_metric/tunelabel_{tune_val_label}_trainer_{trainer_id}/test_metric.pkl")        
    
    # >>>>>>>>>>>>>>>>>>>> Start test inference
    test_time, test_metric = basic_exec_link.test(new_model, config, input_data, 
                                  device = "cpu", checkpoint_file_path = checkpoint_file_path)
    
    print(f"Test Runtime: {test_time:.2f}s")
    
    # <<<<<<<<<<<<<<<<<<<<<< Save the train loss and validation metrics
    
    utils.save_info_pickle(data_path = test_metric_path, target_data = test_metric)
    
    # <<<<<<<<<<<<<<<<<<<<<< Save the profiling info during training
    test_profile_path = os.path.join(workdir,
                        f"tune_{tune_param_name}/test_profile/tunelabel_{tune_val_label}_trainer_{trainer_id}/test_profile.pkl")
    
    utils.save_info_pickle(data_path = test_profile_path, target_data = {"test_time" : test_time})                                          




#### ============================ For classification prediction  ========================

def execute_train_classification(data_name, model_emb, input_data, workdir, config, 
                              tune_param_name, tune_val_label, tune_val, trainer_id = 0, device = "cpu"):
    """
        Train wrapper for tuning hyper-parameters during training
            1) May include a list of tuning values
            2) for each training value, include repeated experiments, multi-trainer_id

        Args:
            data_file_path : Already pre-processed input dataset
            workdir : code execution main directory
            config : yaml file containing main settings
    """
    # >>>>>>>>>>>>>>>>>>>> establish a new pprgo model, set up the model checkpoint for best validation metrics
    checkpoint_file_path = os.path.join(workdir,
                                f"tune_{tune_param_name}/model_checkpoint/tunelabel_{tune_val_label}_trainer_{trainer_id}/best_model.pkl")

    if os.path.exists(os.path.dirname(checkpoint_file_path)):
        print("ckpt folder already exists, so removed ...")
        shutil.rmtree(os.path.dirname(checkpoint_file_path))
    os.makedirs(os.path.dirname(checkpoint_file_path), exist_ok=True)

    # ==========================  Start the training ==========================
    val_metric_path = os.path.join(workdir, 
                                f"tune_{tune_param_name}/val_metric/tunelabel_{tune_val_label}_trainer_{trainer_id}/val_metric.pkl")

    time_training, metric_summary = basic_exec_node_classification.train(model_emb, config, input_data, device = device, checkpoint_file_path = checkpoint_file_path)
    
    logging.info(f"Training task (tune_param_name: {tune_param_name}; tune_val: {tune_val}; trainer_id: {trainer_id}) take: {time_training:.3f} seconds")
    # ==========================  End of the training ==========================
    
    # <<<<<<<<<<<<<<<<<<<<<< Save the train loss and validation metrics
    
    utils.save_info_pickle(data_path = val_metric_path, target_data = metric_summary )
    
    # <<<<<<<<<<<<<<<<<<<<<< Save the profiling info during training
    
    train_profile_folder = os.path.join(workdir, f"tune_{tune_param_name}/train_profile/tunelabel_{tune_val_label}_trainer_{trainer_id}/")                       
    train_profile_path = os.path.join(train_profile_folder, "train_profile.pkl")
    
    utils.save_info_pickle(data_path = train_profile_path, target_data = {"time_training" : time_training})  
    

def execute_test_classification(data_name, input_data, workdir, config, 
                             tune_param_name, tune_val_label, tune_val, trainer_id = 0, device = "cpu"):
    """
        Train wrapper for tuning hyper-parameters during training
            1) May include a list of tuning values
            2) for each training value, include repeated experiments, multi-trainer_id
    """
    # >>>>>>>>>>>>>>>>>>>> establish a new pprgo model, set up the model checkpoint for best validation metrics
    checkpoint_file_path = os.path.join(workdir,
                                f"tune_{tune_param_name}/model_checkpoint/tunelabel_{tune_val_label}_trainer_{trainer_id}/best_model.pkl")

    if not os.path.exists(os.path.dirname(checkpoint_file_path)):
        raise("checkpoint file is missing")

    test_metric_path = os.path.join(workdir,
                                f"tune_{tune_param_name}/test_metric/tunelabel_{tune_val_label}_trainer_{trainer_id}/test_metric.pkl")        
    
    # >>>>>>>>>>>>>>>>>>>> Start test inference
    test_time, test_metric = basic_exec_node_classification.test(config, input_data, 
                                  device = device, checkpoint_file_path = checkpoint_file_path)
    
    print(f"Test Runtime: {test_time:.2f}s")
    
    # <<<<<<<<<<<<<<<<<<<<<< Save the train loss and validation metrics
    
    utils.save_info_pickle(data_path = test_metric_path, target_data = test_metric)
    
    # <<<<<<<<<<<<<<<<<<<<<< Save the profiling info during training
    test_profile_path = os.path.join(workdir,
                        f"tune_{tune_param_name}/test_profile/tunelabel_{tune_val_label}_trainer_{trainer_id}/test_profile.pkl")
    
    utils.save_info_pickle(data_path = test_profile_path, target_data = {"test_time" : test_time})    