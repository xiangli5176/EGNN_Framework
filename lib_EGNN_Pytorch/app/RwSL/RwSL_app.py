from ...app_interface.Graph_Emb_interface import Graph_Emb
from . import basic_exec_cluster


class RwSL_framework(Graph_Emb):
    """ Unified framework based on the RwCL training algorithm for multiple learning tasks.

    Args:
        Graph_Emb : interface
    """
    def __init__(self, config):
        self.config = config
        
        
    def train_cluster(self, model_train, config, inputs, 
            device = "cpu", checkpoint_file_path = None):
        """
        Args:
            model_train: established model to be trained
            config: configuration dict for the hyper-parameter settings
            inputs: [feature, y, adj], 
                    feature: numpy.array, node attributes
                    y: numpy.array , golden label for classes
                    adj : pytorch sparse tensor
            checkpoint_file_path : keep snapshot of the model with the best pairwise f1 score for clustering
        """
        
        return basic_exec_cluster.train(model_train, config, inputs, 
                    device = device, checkpoint_file_path = checkpoint_file_path)
        
    
    def test_cluster(self, model_eval, config, inputs, 
         device = "cpu", checkpoint_file_path = None):
        """
        Evaluate the trained model on the test data
        Args:
            model_template : blank dmon model
            inputs: [feature, y, adj], 
                    feature: numpy.array, node attributes
                    y: numpy.array , golden label for classes
                    adj : pytorch sparse tensor
            checkpoint_file_path : the path for the file of the saved model checkpoint
        """
        
        return basic_exec_cluster.test(model_eval, config, inputs, 
                    device = device, checkpoint_file_path = checkpoint_file_path)
        
        
    def pretrain_ae(self, model_pretrain, config, feature,
                    device = "cpu", pretrain_save_path = None):
        """
        Args:
            model_pretrain: established AE model to be trained
            config: configuration dict for the hyper-parameter settings
            dataset: 
                    feature : is the raw feature
            pretrain_save_path : Save the pre-trained AE model parameter to this Path
        Return: 
            train_time (float) 
            loss_hist (dict) : train loss of pre-train
        """
        
        return basic_exec_cluster.pretrain_ae(model_pretrain, config, feature,
                device = device, pretrain_save_path = pretrain_save_path)