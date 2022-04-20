import pickle
import scipy.sparse as sp
import sklearn.preprocessing as skp
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import csv
from collections import defaultdict


def generate_data_name_train_time_profile(data_name_list, workdir, 
                                     tune_param_name, tune_val_label, trainer_id_list):    
    data_time_dict = {}
    for data_name in data_name_list:
        train_time_list = []
        for trainer_id in trainer_id_list:
            file_path = os.path.join(workdir, f"{data_name}/tune_{tune_param_name}", 
                                    f"train_profile/tunelabel_{tune_val_label}_trainer_{trainer_id}", 
                                    "train_profile.pkl")
#             print(file_path)
            with open(file_path, "rb") as fp:
                train_profile_dict = pickle.load(fp)
                train_time_list.append(train_profile_dict["time_training"])
            
            train_time_array = np.array(train_time_list)
        train_mean, train_std = np.mean(train_time_array), np.std(train_time_array)
        data_time_dict[data_name] = (train_mean, train_std)
    
    target_file = os.path.join(workdir, f"summary/tunelabel_{tune_val_label}/data_name_vs_time.csv")
    os.makedirs(os.path.dirname(target_file), exist_ok = True)
    with open(target_file, 'w', newline='\n') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        header = ["Data Name", "mean", "std"]
        wr.writerow(header)
        for data_name_val, (ave, std) in data_time_dict.items():
            wr.writerow([data_name_val, ave, std])

def generate_size_train_time_profile(data_name, workdir, node_size, 
                                     tune_param_name, tune_val_label, trainer_id_list):    
    size_time_dict = {}
    for tag in node_size:
        train_time_list = []
        for trainer_id in trainer_id_list:
            file_path = os.path.join(workdir, f"{data_name}_{tag}/tune_{tune_param_name}", 
                                    f"train_profile/tunelabel_{tune_val_label}_trainer_{trainer_id}", 
                                    "train_profile.pkl")
#             print(file_path)
            with open(file_path, "rb") as fp:
                train_profile_dict = pickle.load(fp)
                train_time_list.append(train_profile_dict["time_training"])
            
            train_time_array = np.array(train_time_list)
        train_mean, train_std = np.mean(train_time_array), np.std(train_time_array)
        size_time_dict[tag] = (train_mean, train_std)
    
    target_file = os.path.join(workdir, f"summary/tunelabel_{tune_val_label}/size_vs_time.csv")
    os.makedirs(os.path.dirname(target_file), exist_ok = True)
    with open(target_file, 'w', newline='\n') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        header = ["Node Number", "mean", "std"]
        wr.writerow(header)
        for node_size_val, (ave, std) in size_time_dict.items():
            wr.writerow([node_size_val, ave, std])


def plot_stats_test_table(workdir, tune_param_name, tune_val_label_list, tune_val_list, trainer_id_list, skip_trainer = None, 
                    xlabel_text = None):
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
    
    file_name_dict = {"f1_macro" : "f1_macro_tune.csv", "f1_micro" : "f1_micro_tune.csv", "accuracy" : "f1_micro_tune.csv", 
                      "NMI" : "nmi_tune.csv", 
                      "ARI" : "ari_tune.csv",
                      "conductance" : "conductance_tune.csv", "modularity" : "modularity_tune.csv",
                      }
    
    mean_metric = {metric_name : list(pd.read_csv(os.path.join(stats_metric_folder, file_name))["mean"]) 
            for metric_name, file_name in file_name_dict.items()}           
    mean_metric[tune_param_name] = list(pd.read_csv(os.path.join(stats_metric_folder, file_name_dict["f1_micro"]))[tune_param_name])
    
    mean_stats_df = pd.DataFrame.from_dict(mean_metric)
    mean_stats_df.set_index(mean_stats_df[tune_param_name], inplace=True)
    
    plt.clf()
    sns.set_style("whitegrid")
    plt.figure()
    with pd.plotting.plot_params.use('x_compat', True):
        mean_stats_df['f1_macro'].plot(color = 'r', style='+-')
        mean_stats_df['accuracy'].plot(color = 'g', linestyle='-', marker='o')
        # mean_stats_df['f1_micro'].plot(color = 'g', linestyle='-', marker='o')
        mean_stats_df['NMI'].plot(color = 'c', style='*-')
        mean_stats_df['ARI'].plot(color = 'b', style='.--')
        mean_stats_df['conductance'].plot(color = 'g', style='.--')
        mean_stats_df['modularity'].plot(color = 'r', style='*-')

    
    plt.legend()
    xlabel_name = tune_param_name if xlabel_text is None else xlabel_text
    plt.title(f"Clustering Metrics VS {xlabel_name}")
    
    plt.xlabel(xlabel_name)
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
    
    file_name_list = ["f1_macro_tune.csv", "nmi_tune.csv", "f1_micro_tune.csv", 
                        "ari_tune.csv", "conductance_tune.csv", "modularity_tune.csv", 
                    ]
    f1_macro_raw_df, nmi_raw_df, f1_micro_raw_df, ari_raw_df, conductance_raw_df, modularity_raw_df = \
                [pd.read_csv(os.path.join(test_metric_folder, file_name)) for file_name in file_name_list]    
    
    plt.clf()
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharey=False)

    sns.boxplot(ax = axes[0, 0], x=tune_param_name, y="f1_macro",  data=f1_macro_raw_df)
    sns.boxplot(ax = axes[0, 1], x=tune_param_name, y="NMI",  data=nmi_raw_df)
    sns.boxplot(ax = axes[0, 2], x=tune_param_name, y="f1_micro",  data=f1_micro_raw_df)
    sns.boxplot(ax = axes[1, 0], x=tune_param_name, y="ARI",  data=ari_raw_df)
    sns.boxplot(ax = axes[1, 1], x=tune_param_name, y="conductance",  data=conductance_raw_df)
    sns.boxplot(ax = axes[1, 2], x=tune_param_name, y="modularity",  data=modularity_raw_df)
    
    fig.suptitle(f"Clustering metrics VS {tune_param_name}")
    img_file_path = os.path.join(test_metric_folder, f"Summary_raw_tune_{tune_param_name}.pdf")
    plt.savefig(img_file_path, bbox_inches='tight')


def generate_val_all_metric(workdir, tune_param_name, tune_val_label_list, tune_val_list, trainer_id_list, real_time = False):
    sns.set_style("whitegrid")

    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_id_list:
            optimum_val_metric_folder = os.path.join(workdir,
                                f"tune_{tune_param_name}/val_metric/tunelabel_{tune_val_label}_trainer_{trainer_id}/")
            
            optimum_val_metric_path = os.path.join(optimum_val_metric_folder, f"Summary_val_all_metrics.csv")
            
            optimum_val_table = pd.read_csv(optimum_val_metric_path)
            
            metric_epoch_table = optimum_val_table[["f1_macro", "f1_micro", "NMI", "ARI", "conductance", "modularity"]]

            if real_time:
                metric_epoch_table.set_index(optimum_val_table["Time_train"], inplace=True)
            else:
                metric_epoch_table.set_index(optimum_val_table["Epoch_train"], inplace=True)

            # ================= plot all the validation metrics
            plt.figure()
            with pd.plotting.plot_params.use('x_compat', True):
                metric_epoch_table['f1_macro'].plot(color = 'g')
                metric_epoch_table['f1_micro'].plot(color = 'r')
                metric_epoch_table['NMI'].plot(color = 'c')
                metric_epoch_table['ARI'].plot(color = 'b')
                metric_epoch_table['conductance'].plot(color = 'y')
                metric_epoch_table['modularity'].plot(color = 'm')

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


def generate_val_optimum_table(workdir, tune_param_name, tune_val_label_list, tune_val_list, trainer_id_list):
    """
    Draw figures from the validation metrics
    Basic structure of the loading metrics during training is:
        optimum_val_metric = {"f1_macro" : max(val_f1_macro_hist.values()), "NMI" : max(val_nmi_hist.values()), 
                    "ari" : max(val_ari_hist.values()), "f1_micro" : max(val_f1_micro_hist.values())}
    """
    f1_macro_tune_dict = {}
    f1_micro_tune_dict = {}
    nmi_tune_dict = {}
    ari_tune_dict = {}
    conductance_tune_dict = {}
    modularity_tune_dict = {}
    
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_id_list:
            optimum_val_metric_path = os.path.join(workdir,
                                f"tune_{tune_param_name}/val_metric/tunelabel_{tune_val_label}_trainer_{trainer_id}/Optimum_val_all_metrics.pkl")

            with open(optimum_val_metric_path, "rb") as fp:
                optimum_val_metric = pickle.load(fp)
                print(f" tune_param_name : {tune_param_name} | tune_val: {tune_val} | trainer_id: {trainer_id} | {optimum_val_metric}")
                f1_macro_tune_dict[(tune_val, trainer_id)] = optimum_val_metric["f1_macro"]
                f1_micro_tune_dict[(tune_val, trainer_id)] = optimum_val_metric["f1_micro"]
                ari_tune_dict[(tune_val, trainer_id)] = optimum_val_metric["ARI"]
                nmi_tune_dict[(tune_val, trainer_id)] = optimum_val_metric["NMI"]
                conductance_tune_dict[(tune_val, trainer_id)] = optimum_val_metric["conductance"]
                modularity_tune_dict[(tune_val, trainer_id)] = optimum_val_metric["modularity"]

    optimum_val_metric_folder = os.path.join(workdir,
                                    f"tune_{tune_param_name}/val_metric/optimum_val/")
    os.makedirs(os.path.dirname(optimum_val_metric_folder), exist_ok=True)

    generate_tuning_raw_data_table(f1_macro_tune_dict, optimum_val_metric_folder, 
                                file_name="optimum_val_f1_macro_tune.csv", tune_param_name = tune_param_name, metric_name = "f1_macro")
    
    generate_tuning_raw_data_table(f1_micro_tune_dict, optimum_val_metric_folder, 
                                file_name="optimum_val_f1_micro_tune.csv", tune_param_name = tune_param_name, metric_name = "f1_micro")

    generate_tuning_raw_data_table(ari_tune_dict, optimum_val_metric_folder, 
                                file_name="optimum_val_ari_tune.csv", tune_param_name = tune_param_name, metric_name = "ARI")

    generate_tuning_raw_data_table(nmi_tune_dict, optimum_val_metric_folder, 
                                file_name="optimum_val_nmi_tune.csv", tune_param_name = tune_param_name, metric_name = "NMI")

    generate_tuning_raw_data_table(conductance_tune_dict, optimum_val_metric_folder, 
                                file_name="optimum_val_conductance_tune.csv", tune_param_name = tune_param_name, metric_name = "conductance")

    generate_tuning_raw_data_table(modularity_tune_dict, optimum_val_metric_folder, 
                                file_name="optimum_val_modularity_tune.csv", tune_param_name = tune_param_name, metric_name = "modularity")


def generate_test_epoch_table(workdir, tune_param_name, tune_val_label_list, tune_val_list, trainer_id_list, skip_trainer = None):
    """
        generate a table indicating the test model (with the best clustering metric performance) epoch
    """
    test_time_dict = {}
    test_epoch_dict = {}
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_id_list:
            test_epoch_file_path = os.path.join(workdir,
                                f"tune_{tune_param_name}/model_checkpoint/tunelabel_{tune_val_label}_trainer_{trainer_id}/test_epoch.csv")

            if not os.path.exists(os.path.dirname(test_epoch_file_path)):
                raise("checkpoint file is missing")
            
            with open(test_epoch_file_path, newline='\n') as f:
                reader = csv.reader(f)
#                 test_epoch_num = [int(row[-1]) for row in reader]
                test_epoch_num, test_time_val = next(reader)[-2:]
                test_epoch_num = int(test_epoch_num)
                test_time_val = float(test_time_val)
#                 print(type(test_epoch_num), test_epoch_num)
                test_epoch_dict[(tune_val, trainer_id)] = test_epoch_num
                test_time_dict[(tune_val, trainer_id)] = test_time_val
                
    test_metric_sub_folder = "figure_tune/no_skip_trainer/" if not skip_trainer else "figure_tune/skip_trainer/"
    test_metric_folder = os.path.join(workdir, f"tune_{tune_param_name}/test_metric/", test_metric_sub_folder)
    os.makedirs(os.path.dirname(test_metric_folder), exist_ok=True)
    
    generate_tuning_raw_data_table(test_epoch_dict, test_metric_folder, 
                file_name="test_epoch_num.csv", tune_param_name = tune_param_name, metric_name = "test_epoch_num", skip_trainer = skip_trainer)
                
    generate_tuning_raw_data_table(test_time_dict, test_metric_folder, 
                file_name="test_time_value.csv", tune_param_name = tune_param_name, metric_name = "test_time_val", skip_trainer = skip_trainer)


def generate_test_table(workdir, tune_param_name, tune_val_label_list, tune_val_list, trainer_id_list, skip_trainer = None):
    """
    Draw figures from the test metrics
    Basic structure of the loading metrics during training is:
        test_metric = {"micro_f1_train" : micro_f1_train, "micro_f1_val" : micro_f1_val, "micro_f1_test" : micro_f1_test}
    Args:
        skip_trainer (iterable) : list or set of trainer we want to skip as outliers
    """
    f1_macro_tune_dict = {}
    f1_micro_tune_dict = {}
    nmi_tune_dict = {}
    ari_tune_dict = {}
    conductance_tune_dict = {}
    modularity_tune_dict = {}

    
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_id_list:
            test_metric_path = os.path.join(workdir,
                                        f"tune_{tune_param_name}/test_metric/tunelabel_{tune_val_label}_trainer_{trainer_id}/test_metric.pkl")

            with open(test_metric_path, "rb") as fp:
                test_metric = pickle.load(fp)
                print(f"tune_param_name : {tune_param_name} | tune_val: {tune_val} | trainer_id: {trainer_id} | {test_metric}")
                f1_macro_tune_dict[(tune_val, trainer_id)] = test_metric["f1_macro"]
                f1_micro_tune_dict[(tune_val, trainer_id)] = test_metric["f1_micro"]
                nmi_tune_dict[(tune_val, trainer_id)] = test_metric["NMI"]
                ari_tune_dict[(tune_val, trainer_id)] = test_metric["ARI"]
                conductance_tune_dict[(tune_val, trainer_id)] = test_metric["conductance"]
                modularity_tune_dict[(tune_val, trainer_id)] = test_metric["modularity"]

    
    test_metric_sub_folder = "figure_tune/no_skip_trainer/" if not skip_trainer else "figure_tune/skip_trainer/"
        
    test_metric_folder = os.path.join(workdir, f"tune_{tune_param_name}/test_metric/", test_metric_sub_folder)

    os.makedirs(os.path.dirname(test_metric_folder), exist_ok=True)
    
    generate_tuning_raw_data_table(f1_macro_tune_dict, test_metric_folder, 
                        file_name="f1_macro_tune.csv", tune_param_name = tune_param_name, metric_name = "f1_macro", skip_trainer = skip_trainer)
    
    generate_tuning_raw_data_table(f1_micro_tune_dict, test_metric_folder, 
                        file_name="f1_micro_tune.csv", tune_param_name = tune_param_name, metric_name = "f1_micro", skip_trainer = skip_trainer)

    generate_tuning_raw_data_table(nmi_tune_dict, test_metric_folder, 
                    file_name="nmi_tune.csv", tune_param_name = tune_param_name, metric_name = "NMI", skip_trainer = skip_trainer)


    generate_tuning_raw_data_table(ari_tune_dict, test_metric_folder, 
                        file_name="ari_tune.csv", tune_param_name = tune_param_name, metric_name = "ARI", skip_trainer = skip_trainer)


    generate_tuning_raw_data_table(conductance_tune_dict, test_metric_folder, 
                        file_name="conductance_tune.csv", tune_param_name = tune_param_name, metric_name = "conductance", skip_trainer = skip_trainer)

    generate_tuning_raw_data_table(modularity_tune_dict, test_metric_folder, 
                        file_name="modularity_tune.csv", tune_param_name = tune_param_name, metric_name = "modularity", skip_trainer = skip_trainer)

    

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

    # print({key : len(val) for key, val in metric_dict.items()}, "\n")

    val_summary = {"Time_train" : Time_train, "Epoch_train" : Epoch_train,  
                    "f1_macro" : list(metric_dict["f1_macro_hist"].values()), 
                    "f1_micro" : list(metric_dict["f1_micro_hist"].values()), 
                    "NMI" : list(metric_dict["nmi_hist"].values()), 
                    "ARI" : list(metric_dict["ari_hist"].values()), 
                    "conductance" : list(metric_dict["conductance_hist"].values()), 
                    "modularity" : list(metric_dict["modularity_hist"].values()), 
                    }
    
    # generate all the distribution for each epoch
    generate_negative_hardness_plot(Epoch_train, list(metric_dict["semantic_neg_frac_distr_hist"].values()), 
                            file_name = 'negative_hard_distr', save_folder = os.path.dirname(val_metric_path)  )

    val_summary = pd.DataFrame.from_dict(val_summary)
    val_summary.to_csv(os.path.join(os.path.dirname(val_metric_path), f"Summary_val_all_metrics.csv"), index=False)

    val_optimum = {"NMI" : max(metric_dict["nmi_hist"].values()), 
                    "f1_macro" : max(metric_dict["f1_macro_hist"].values()), 
                    "f1_miro" : max(metric_dict["f1_micro_hist"].values()),
                    "ARI" : max(metric_dict["ari_hist"].values()), 
                    "conductance" : max(metric_dict["conductance_hist"].values()), 
                    "modularity" : max(metric_dict["modularity_hist"].values()),
                    }

    with open(os.path.join(os.path.dirname(val_metric_path), f"Optimum_val_all_metrics.pkl"), "wb") as fp:
        pickle.dump(val_optimum, fp)

def generate_negative_hardness_plot(train_epoch_list, semantic_distr_list, title = None, x_label = 'Similarity Score',
                 y_label = 'Sample Pair Fraction', file_name = 'default', save_folder = ''):        
    """ generate the similarity score for all the negative sample pairs

    Args:
        train_epoch ([type]): [description]
        semantic_distr ([type]): [description]
        title ([type], optional): [description]. Defaults to None.
        y_label (str, optional): [description]. Defaults to 'Properties'.
        file_name (str, optional): [description]. Defaults to 'default'.
        save_folder (str, optional): [description]. Defaults to ''.
    """
    
    for train_epoch, semantic_distr in zip(train_epoch_list, semantic_distr_list):
        hist, bin_edges = semantic_distr
        save_file_name = os.path.join(save_folder, f'{file_name}_epoch_{train_epoch}.pdf' ) 

        generate_negative_hardness_distr(hist, bin_edges, save_file_name, 
                        title = title, x_label = x_label, y_label = y_label)


def generate_negative_hardness_distr(hist, bin_edges, save_file_name, 
                        title = None, x_label = 'Similarity Score', y_label = 'Sample Pair Fraction'):
    fig, ax = plt.subplots()
    # np.diff: Calculate the n-th discrete difference along the given axis.
    ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="black", align="edge") 
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    plt.savefig(save_file_name)        


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


def generate_tsne_numpy(numpy_emb, labels, save_path_folder = None, file_name = None,
                       x_label = None, save_file = True):
    u, s, v = sp.linalg.svds(numpy_emb, k=16, which='LM')
    pca_emb_np = skp.normalize(numpy_emb.dot(v.T))

    feat_cols = [ 'feat'+str(i) for i in range(pca_emb_np.shape[1]) ]
    pca_emb_df = pd.DataFrame(pca_emb_np, columns=feat_cols)
    pca_emb_df['y'] = labels

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(pca_emb_df[feat_cols].values)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    sns.set_style("white")
    pca_emb_df['tsne-2d-one'] = tsne_results[:,0]
    pca_emb_df['tsne-2d-two'] = tsne_results[:,1]
#     plt.figure(figsize=(16,10))
    g = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
    #     palette=sns.color_palette("hls", 7),
        palette=sns.color_palette("bright", pca_emb_df['y'].nunique()),
        data=pca_emb_df,
        legend="",
        alpha=0.3
    )
    g.set(xticklabels=[], yticklabels=[],  ylabel=None)
    if x_label is not None:
        plt.xlabel(x_label, size = 14)
    else:
        g.set(xlabel=None)

    sns.despine( left=True, bottom=True)
    
    if save_file:
        img_file_path = os.path.join(save_path_folder, file_name)
        os.makedirs(os.path.dirname(img_file_path), exist_ok=True)
        plt.savefig(img_file_path, bbox_inches='tight')


def generate_test_tsne(labels, workdir, tune_param_name, tune_val_label_list, tune_val_list, trainer_id_list):
    """
    Generate 2D t-SNE to visualize the clustering potential of the embedding
    labels: golden labels 
    """
    
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_id_list:
            test_emb_path = os.path.join(workdir,
                    f"tune_{tune_param_name}/test_metric/tunelabel_{tune_val_label}_trainer_{trainer_id}/test_emb.pkl")

            with open(test_emb_path, "rb") as fp:
                test_emb = pickle.load(fp)
    
            test_tsne_folder = os.path.join(workdir, f"tune_{tune_param_name}/test_metric/", f"tSNE/tunelabel_{tune_val_label}_trainer_{trainer_id}/") 
            os.makedirs(os.path.dirname(test_tsne_folder), exist_ok=True)

            generate_tsne_numpy(test_emb, labels, test_tsne_folder, file_name = "Test_tSNE.pdf", x_label = None, save_file = True)
            
            
def generate_val_tsne(labels, workdir, tune_param_name, tune_val_label_list, tune_val_list, trainer_id_list):
    """
    Generate 2D t-SNE to visualize the clustering potential of the embedding
    labels: golden labels 
    """
    for tune_val_label, tune_val in zip(tune_val_label_list, tune_val_list):
        for trainer_id in trainer_id_list:
            val_emb_path = os.path.join(workdir,
                    f"tune_{tune_param_name}/val_metric/tunelabel_{tune_val_label}_trainer_{trainer_id}/val_emb.pkl")

            val_tsne_folder = os.path.join(workdir, f"tune_{tune_param_name}/val_metric/", f"tSNE/tunelabel_{tune_val_label}_trainer_{trainer_id}/") 
            os.makedirs(os.path.dirname(val_tsne_folder), exist_ok=True)
            
            with open(val_emb_path, "rb") as fp:
                val_emb = pickle.load(fp)     
            
            col_num = len(val_emb)
            height = 5
            fig = plt.figure( figsize=(height * col_num, height) )
            i = 0
            for (real_time, epoch), val_emb_np in val_emb.items():
                i += 1
                ax = fig.add_subplot(1, col_num, i)
                generate_tsne_numpy(val_emb_np, labels, val_tsne_folder, 
                        file_name = f"val_tSNE_epoch_{epoch}.pdf", x_label = None, save_file = False)
            
            
            img_file_path = os.path.join(val_tsne_folder, "Val_tSNE_seq.pdf")
            plt.savefig(img_file_path, bbox_inches='tight')    