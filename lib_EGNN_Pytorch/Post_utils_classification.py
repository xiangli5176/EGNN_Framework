import pickle
import scipy.sparse as sp
import sklearn.preprocessing as skp
import time
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import csv
from collections import defaultdict


### ========================== These 3 functions below is to prepare validation data results in csv ================================
def draw_val_metrics(workdir, tune_param_name, tune_val_label, trainer_id = 0):
    """
    Draw figures from the validation metrics
    Here the accuracy and f1_micro_score are using interchangeably
    Each element of these dict the nested structure:  {(seconds, epoch) : metric value}
    """
    val_metric_path = os.path.join(workdir,
                                f"tune_{tune_param_name}/val_metric/tunelabel_{tune_val_label}_trainer_{trainer_id}/val_metric.pkl")
    
    
    with open(val_metric_path, "rb") as fp:
        metric_dict = pickle.load(fp)
    
    # <<<<<<<<<<<<<<< Generate a summary table for all metrics
    Time_train = [seconds for (seconds, epoch) in metric_dict["loss_hist"].keys()]
    Epoch_train = [epoch for (seconds, epoch) in metric_dict["loss_hist"].keys()]

    ### <<<<<<<<<<<< Generate training loss VS Time (s)
    generate_val_table(metric_dict["loss_hist"], os.path.dirname(val_metric_path), file_name="train_loss_hist.csv", metric_name = "loss")
    train_loss_time = {seconds : val for (seconds, epoch), val in metric_dict["loss_hist"].items()}
    generate_val_figure(train_loss_time, os.path.dirname(val_metric_path), file_name="train_loss.pdf", 
                        img_title = "Train Loss VS Time (s)", xlabel = "Training Time (s)", ylabel = "Train Loss")

    val_summary = {"Time_train" : Time_train, "Epoch_train" : Epoch_train,  
                    "f1_micro" : list(metric_dict["f1_micro_hist"].values()), 
                    "accuracy" : list(metric_dict["accuracy_hist"].values()), 
                    "f1_macro" : list(metric_dict["f1_macro_hist"].values()), 
                    }
    
    val_summary = pd.DataFrame.from_dict(val_summary)
    val_summary.to_csv(os.path.join(os.path.dirname(val_metric_path), f"Summary_val_all_metrics.csv"), index=False)

    val_optimum = {"f1_micro" : max(metric_dict["f1_micro_hist"].values()), 
                    "accuracy" : max(metric_dict["accuracy_hist"].values()), 
                    "f1_macro" : max(metric_dict["f1_macro_hist"].values()), 
                    }

    with open(os.path.join(os.path.dirname(val_metric_path), f"Optimum_val_all_metrics.pkl"), "wb") as fp:
        pickle.dump(val_optimum, fp)



def generate_val_table(data_dict, file_path, file_name, metric_name):
    """
        data_dict : a dictionary of different runing index with different tuning values
                data_dict[1]: e.g.  index 1 runing, this is a dictionary of tuning values
    """
    target_file = os.path.join(file_path, file_name)
    with open(target_file, 'w', newline='\n') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        header = ["Time(s)", "epoch", metric_name]
        wr.writerow(header)
        for (seconds, epoch), val in data_dict.items():
            tmp_line = [seconds, epoch, val]
            wr.writerow(tmp_line)
            
            
def generate_val_figure(data_dict, file_path, file_name, img_title, xlabel, ylabel):
    plt.clf()
    plt.figure()
    sns.set(style='whitegrid')

    figure_data = {}
    figure_data[xlabel] = sorted(data_dict.keys())
    figure_data[ylabel] = [data_dict[key] for key in figure_data[xlabel]]
    df = pd.DataFrame(figure_data) 

    g = sns.relplot(x = xlabel, y = ylabel, markers=True, kind="line", data=df)
    g.despine(left=True)
    g.fig.suptitle(img_title)
    g.set_xlabels(xlabel)
    g.set_ylabels(ylabel)

    img_file_path = os.path.join(file_path, file_name)
    plt.savefig(img_file_path, bbox_inches='tight')


#### ============================= Generate the validation results figures =======================
def generate_val_all_metric(workdir, tune_param_name, tune_val_label_list, tune_val_list, trainer_id_list, real_time = False):
    sns.set_style("whitegrid")

    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_id_list:
            optimum_val_metric_folder = os.path.join(workdir,
                                f"tune_{tune_param_name}/val_metric/tunelabel_{tune_val_label}_trainer_{trainer_id}/")
            
            optimum_val_metric_path = os.path.join(optimum_val_metric_folder, f"Summary_val_all_metrics.csv")
            
            optimum_val_table = pd.read_csv(optimum_val_metric_path)
            
            metric_epoch_table = optimum_val_table[["f1_micro", "f1_macro", "accuracy"]]

            if real_time:
                metric_epoch_table.set_index(optimum_val_table["Time_train"], inplace=True)
            else:
                metric_epoch_table.set_index(optimum_val_table["Epoch_train"], inplace=True)

            # ================= plot all the validation metrics
            plt.figure()
            with pd.plotting.plot_params.use('x_compat', True):
                metric_epoch_table['f1_micro'].plot(color = 'g')
                metric_epoch_table['f1_macro'].plot(color = 'r')
                # metric_epoch_table['accuracy'].plot(color = 'b')

            plt.legend()
            plt.title(f'Metrics VS Epoch')
            if real_time:
                plt.xlabel("Train Time (s)")
            else:
                plt.xlabel("Train Epoch")

            plt.ylabel("Clustering Metric")
            save_file_name = f"Summary_val_metric_realtime.pdf" if real_time else f"Summary_val_metric_epoch.pdf" 
            plt.savefig( os.path.join(optimum_val_metric_folder, save_file_name) )


def generate_validation_plot(pd_col_list, color_list, real_time = True, 
                title = None, y_label = 'Properties', file_name = 'default', save_folder = ''):
    plt.figure()
    with pd.plotting.plot_params.use('x_compat', True):
        for column, color in zip(pd_col_list, color_list):
            column.plot(color = color)

    plt.legend()
    if title:
        plt.title(title)

    if real_time:
        plt.xlabel("Train Time (s)")
    else:
        plt.xlabel("Train Epoch")

    plt.ylabel(y_label)
    save_file_name = f'{file_name}_realtime.pdf' if real_time else f'{file_name}_epoch.pdf' 
    plt.savefig( os.path.join(save_folder, save_file_name) )    


#### ============================= Generate test results to csv  =======================
def generate_test_table(workdir, tune_param_name, tune_val_label_list, tune_val_list, trainer_id_list, skip_trainer = None):
    """
    Draw figures from the test metrics
    Basic structure of the loading metrics during training is:
        test_metric = {"micro_f1_train" : micro_f1_train, "micro_f1_val" : micro_f1_val, "micro_f1_test" : micro_f1_test}
    Args:
        skip_trainer (iterable) : list or set of trainer we want to skip as outliers
    """
    f1_micro_tune_dict = {}
    f1_macro_tune_dict = {}
    
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_id_list:
            test_metric_path = os.path.join(workdir,
                            f"tune_{tune_param_name}/test_metric/tunelabel_{tune_val_label}_trainer_{trainer_id}/test_metric.pkl")

            with open(test_metric_path, "rb") as fp:
                test_metric = pickle.load(fp)
                print(f"tune_param_name : {tune_param_name} | tune_val: {tune_val} | trainer_id: {trainer_id} | {test_metric}")
                f1_micro_tune_dict[(tune_val, trainer_id)] = test_metric["f1_micro"]
                f1_macro_tune_dict[(tune_val, trainer_id)] = test_metric["f1_macro"]

    
    test_metric_sub_folder = "figure_tune/no_skip_trainer/" if not skip_trainer else "figure_tune/skip_trainer/"
    test_metric_folder = os.path.join(workdir, f"tune_{tune_param_name}/test_metric/", test_metric_sub_folder)
    os.makedirs(os.path.dirname(test_metric_folder), exist_ok=True)
    
    generate_tuning_raw_data_table(f1_micro_tune_dict, test_metric_folder, 
                        file_name="f1_micro_tune.csv", tune_param_name = tune_param_name, metric_name = "f1_micro", skip_trainer = skip_trainer)
    
    generate_tuning_raw_data_table(f1_macro_tune_dict, test_metric_folder, 
                        file_name="f1_macro_tune.csv", tune_param_name = tune_param_name, metric_name = "f1_macro", skip_trainer = skip_trainer)



def generate_tuning_raw_data_table(data_dict, file_path, file_name, tune_param_name, metric_name, 
                                    skip_trainer = None):
    """
        data_dict : a dictionary of different runing index with different tuning values
                data_dict[1]: e.g.  index 1 runing, this is a dictionary of tuning values
    """
    if skip_trainer is not None:
        skip_trainer = set(skip_trainer)

    target_file = os.path.join(file_path, file_name)
    stats_file = os.path.join(file_path, "metric_stats/", file_name)  # file to record stats of each  metric
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)

    stats_among_trainer = defaultdict(list)

    with open(target_file, 'w', newline='\n') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        header = [tune_param_name, "trainer_id", metric_name]
        wr.writerow(header)
        for (tune_param_val, trainer_id), metric_val in data_dict.items():
            if skip_trainer and trainer_id in skip_trainer:
                continue
            tmp_line = [tune_param_val, trainer_id, metric_val]
            wr.writerow(tmp_line)
            stats_among_trainer[tune_param_val].append(metric_val)

    with open(stats_file, 'w', newline='\n') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        stats_header = [tune_param_name, "mean", "std"]
        wr.writerow(stats_header)
        for tune_param_val, val_list in stats_among_trainer.items():
            if len(val_list) > 4:
                # tmp_line = [tune_param_val, np.mean(sorted(val_list)[1:-1] ), np.std(sorted(val_list)[1:-1])]
                # since there are multi-metrics, hard to abondon the outlier values...
                tmp_line = [tune_param_val, np.mean(sorted(val_list) ), np.std(sorted(val_list))]
            else:
                tmp_line = [tune_param_val, np.mean(val_list), np.std(val_list)]
            wr.writerow(tmp_line)


#### ============================= Plot the test results regarding tuning prarmeters  =======================
def plot_raw_tune_test_table(workdir, tune_param_name, tune_val_label_list, tune_val_list, trainer_id_list, skip_trainer = None):
    """
    Draw figures from the test metrics
    Basic structure of the loading metrics during training is:
        test_metric = {"micro_f1_train" : micro_f1_train, "micro_f1_val" : micro_f1_val, "micro_f1_test" : micro_f1_test}
    Args:
        skip_trainer (iterable) : list or set of trainer we want to skip as outliers
    """

    test_metric_sub_folder = "figure_tune/no_skip_trainer/" if not skip_trainer else "figure_tune/skip_trainer/"
        
    test_metric_folder = os.path.join(workdir, f"tune_{tune_param_name}/test_metric/", test_metric_sub_folder)
    
    os.makedirs(os.path.dirname(test_metric_folder), exist_ok=True)
    
    file_name_list = ["f1_micro_tune.csv", "f1_macro_tune.csv"]
    f1_micro_raw_df, f1_macro_raw_df = [pd.read_csv(os.path.join(test_metric_folder, file_name)) for file_name in file_name_list]    
    
    plt.clf()
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)

    sns.boxplot(ax = axes[0], x=tune_param_name, y="f1_micro",  data = f1_micro_raw_df)
    sns.boxplot(ax = axes[1], x=tune_param_name, y="f1_macro",  data = f1_macro_raw_df)
    
    fig.suptitle(f"Clustering metrics VS {tune_param_name}")
    img_file_path = os.path.join(test_metric_folder, f"Summary_raw_tune_{tune_param_name}.pdf")
    plt.savefig(img_file_path, bbox_inches='tight')



#### ============================= Generate test results statistics  =======================
def plot_stats_test_table(workdir, tune_param_name, tune_val_label_list, tune_val_list, trainer_id_list, skip_trainer = None):
    """
    Draw figures from the test metrics
    Basic structure of the loading metrics during training is:
        test_metric = {"micro_f1_train" : micro_f1_train, "micro_f1_val" : micro_f1_val, "micro_f1_test" : micro_f1_test}
    Args:
        skip_trainer (iterable) : list or set of trainer we want to skip as outliers
    """

    test_metric_sub_folder = "figure_tune/no_skip_trainer/" if not skip_trainer else "figure_tune/skip_trainer/"
        
    stats_metric_folder = os.path.join(workdir, f"tune_{tune_param_name}/test_metric/", test_metric_sub_folder, "metric_stats/")
    
    os.makedirs(os.path.dirname(stats_metric_folder), exist_ok=True)
    
    file_name_dict = {"f1_micro" : "f1_micro_tune.csv", "f1_macro" : "f1_macro_tune.csv" }
    
    mean_metric = {metric_name : list(pd.read_csv(os.path.join(stats_metric_folder, file_name))["mean"])
            for metric_name, file_name in file_name_dict.items()}        

    mean_metric[tune_param_name] = list(pd.read_csv(os.path.join(stats_metric_folder, file_name_dict["f1_micro"]))[tune_param_name])
    
    mean_stats_df = pd.DataFrame.from_dict(mean_metric)
    mean_stats_df.set_index(mean_stats_df[tune_param_name], inplace=True)
    
    plt.clf()
    sns.set_style("whitegrid")
    plt.figure()
    with pd.plotting.plot_params.use('x_compat', True):
        mean_stats_df['f1_micro'].plot(color = 'r', style='+-')
        mean_stats_df['f1_macro'].plot(color = 'g', linestyle='-', marker='o')
    
    plt.legend()
    plt.title(f"Clustering Metrics VS {tune_param_name}")
    plt.xlabel(tune_param_name)
    plt.ylabel("Clustering Metric")
    
    mean_stats_df.to_csv(os.path.join(stats_metric_folder, f"Mean_Value_Stats_metric_{tune_param_name}.csv"), index = True)
    
    plt.savefig(os.path.join(stats_metric_folder, f"Stats_metric_{tune_param_name}.pdf"), bbox_inches='tight')

    # store the information for std error:
    std_metric = {metric_name : list(pd.read_csv(os.path.join(stats_metric_folder, file_name))["std"]) 
                for metric_name, file_name in file_name_dict.items()}           
    std_metric[tune_param_name] = list(pd.read_csv(os.path.join(stats_metric_folder, file_name_dict["f1_micro"]))[tune_param_name])
    std_stats_df = pd.DataFrame.from_dict(std_metric)
    std_stats_df.set_index(std_stats_df[tune_param_name], inplace=True)
    std_stats_df.to_csv(os.path.join(stats_metric_folder, f"Std_Value_Stats_metric_{tune_param_name}.csv"), index = True)




