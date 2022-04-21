


class GNN_framework():
    
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
        
        pass
    
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
        pass
    
    
    def train_link(model_train, config, inputs, 
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
        pass
    
    
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
        pass
    
    
    def train_node_classification(model_emb, config, inputs, device = "cpu", checkpoint_file_path = None):
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
        pass
    
    def test_node_classification(config, inputs, 
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
        pass