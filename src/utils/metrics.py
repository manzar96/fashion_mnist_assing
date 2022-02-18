import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score


class MetricsWraper:
    """
    Class that implements metrics used in BaseTrainer classs.
    """
    def __init__(self, metrics):
        """
        :param metrics: the metrics that you want to be calculated during
        training or validation
        """
        self.avail_metrics = ['accuracy',
         'f1-micro',
         'f1-macro', 'recall-micro',
         'recall-macro']
        asked = [metric for metric in metrics if not metric in \
                                             self.avail_metrics]
        if not len(asked) == 0:
            assert False,"Metrics: {} are not implemented".format(asked)
        self.metrics_names = metrics
        self.num_data = 0
        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []
        self.calc_dict = {}

    def reset(self):
        self.num_data = 0
        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []
        self.calc_dict = {}

    def _compute_metrics_batched(self, outputs, targets, type='train'):
        with torch.no_grad():
            outputs = F.softmax(outputs, dim=-1)
            indexes = torch.argmax(outputs, dim=-1).tolist()
            if type == 'train':
                self.train_preds.append(indexes)
                self.train_targets.append(targets.tolist())

            else:
                self.val_preds.append(indexes)
                self.val_targets.append(targets.tolist())

    def get_metrics(self, type='train'):
        """
        :param type: type of metrics to be reported ('train' or 'val')
        :return: returns a dict containing the specified metrics
        """
        if type == 'train':
            preds = self.train_preds
            targets = self.train_targets
        elif type=="val":
            preds = self.val_preds
            targets = self.val_targets
        else:
            assert False, "MetricsWraper invalid type on get_metrics"
        targets = [elem for sublist in targets for elem in sublist ]
        preds = [elem for sublist in preds for elem in sublist ]
        if 'accuracy' in self.metrics_names:
            self.calc_dict['Accuracy'] = accuracy_score(targets,preds)
        if 'f1-micro' in self.metrics_names:
            self.calc_dict['F1-micro'] = f1_score(targets, preds,
                                                average='micro')
        if 'f1-macro' in self.metrics_names:
            self.calc_dict['F1-macro'] = f1_score(targets, preds,
                                                average='macro')
        if 'recall-micro' in self.metrics_names:
            self.calc_dict['Recall-micro'] = recall_score(targets, preds,
                                                average='micro')
        if 'recall-macro' in self.metrics_names:
            self.calc_dict['Recall-macro'] = recall_score(targets, preds,
                                                average='macro')
        return self.calc_dict
