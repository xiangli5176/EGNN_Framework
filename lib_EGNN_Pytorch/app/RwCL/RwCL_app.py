from ...app_interface.Graph_Emb_interface import Graph_Emb
from . import basic_exec_cluster, basic_exec_link, basic_exec_node_classification


class RwCL_framework(Graph_Emb):
    """ Unified framework based on the RwCL training algorithm for multiple learning tasks.

    Args:
        Graph_Emb : interface
    """
    def __init__(self, config):
        self.config = config
        
    def train_cluster(self, model_train, config, inputs, 
        device = "cpu", checkpoint_file_path = None, 
        metric_list = ["f1_micro", "f1_macro", "Accuracy", "NMI", "ARI", "conductance", "modularity"]):
        """  Main training process for clustering

        Args:
            model_train ([type]): [description]
            config (dict): config of settings
            inputs (list(numpy.array)): feature, labels_full, adj_raw
            device (str, optional): which device to use. Defaults to "cpu".
            checkpoint_file_path ([type], optional): Model state save file. Defaults to None.
            metric_list : metrics to calculate
        Returns:
            tuple(dict): training statistics
        """
        
        return basic_exec_cluster.train(model_train, config, inputs, 
                    device = device, checkpoint_file_path = checkpoint_file_path, 
                    metric_list = metric_list)
        
    
    def test_cluster(self, model_val, config, inputs, 
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
        
        return basic_exec_cluster.test(model_val, config, inputs, 
                    device = device, checkpoint_file_path = checkpoint_file_path)
        
        
    def train_link(self, model_train, config, inputs, 
        device = "cpu", checkpoint_file_path = None, 
        metric_list = ["auc_score", "ap_score"],
        ):
        """  Main training process for clustering

        Args:
            model_train ([type]): [description]
            config (dict): config of settings
            inputs (list(numpy.array)): feature, labels_full, adj_raw
            device (str, optional): which device to use. Defaults to "cpu".
            checkpoint_file_path ([type], optional): Model state save file. Defaults to None.
            metric_list : metrics to calculate
        Returns:
            tuple(dict): training statistics
        """
        
        return basic_exec_link.train(model_train, config, inputs, 
                    device = device, checkpoint_file_path = checkpoint_file_path, 
                    metric_list = metric_list)
        
    
    def test_link(self, model_val, config, inputs, 
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
        
        return basic_exec_link.test(model_val, config, inputs, 
                    device = device, checkpoint_file_path = checkpoint_file_path)
        
        
    def train_node_classification(self, model_train, config, inputs, device = "cpu", checkpoint_file_path = None):
        """ 
        Main procedure to perform train process: 
            1) pre-computation/training for embedding
            2) regression train for supervised classification task
        Args:
            model_train ([type]): [description]
            config (dict): [description]
            inputs (list(numpy.array)): features, labels_full, idx_train, idx_val, idx_test
            device (str, optional): [description]. Defaults to "cpu".
            checkpoint_file_path ([type], optional): [description]. Defaults to None.
        """
        return basic_exec_node_classification.train(model_train, config, inputs, 
                    device = device, checkpoint_file_path = checkpoint_file_path)
    
    def test_node_classification(self, config, inputs, 
                    device = "cpu", checkpoint_file_path = None):
        """ 
        Main procedure to perform train process: 
            1) pre-computation/training for embedding
            2) regression train for supervised classification task
        Args:
            model_train ([type]): [description]
            config (dict): [description]
            inputs (list(numpy.array)): features, labels_full, idx_train, idx_val, idx_test
            device (str, optional): [description]. Defaults to "cpu".
            checkpoint_file_path ([type], optional): [description]. Defaults to None.
        """
        return basic_exec_node_classification.test(config, inputs, 
                    device = device, checkpoint_file_path = checkpoint_file_path)