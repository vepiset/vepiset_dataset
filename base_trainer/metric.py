
import sys
import torch
import numpy as np
import warnings


from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve
from sklearn.metrics import confusion_matrix

sys.path.append('.')


warnings.filterwarnings('ignore')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ROCAUCMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):

        self.y_true_11=None
        self.y_pred_11 = None

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy()

        y_pred = torch.sigmoid(y_pred).data.cpu().numpy()

        if self.y_true_11 is None:
            self.y_true_11 = y_true
            self.y_pred_11 = y_pred
        else:
            self.y_true_11 = np.concatenate((self.y_true_11, y_true),axis=0)
            self.y_pred_11 = np.concatenate((self.y_pred_11, y_pred),axis=0)

        return self.y_true_11, self.y_pred_11

    def fast_auc(self,y_true, y_prob):


        y_true = np.asarray(y_true)
        y_true = y_true[np.argsort(y_prob)]
        cumfalses = np.cumsum(1 - y_true)
        nfalse = cumfalses[-1]
        auc = (y_true * cumfalses).sum()

        auc /= (nfalse * (len(y_true) - nfalse))
        return auc

    @property
    def avg(self):

        self.y_true_11=self.y_true_11.reshape(-1)
        self.y_pred_11 = self.y_pred_11.reshape(-1)
        score=self.fast_auc(self.y_true_11, self.y_pred_11)

        return score

    def evaluate(y_true, y_pred, digits=4, cutoff='auto'):

        if cutoff == 'auto':
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            youden = tpr - fpr
            cutoff = thresholds[np.argmax(youden)]

        return cutoff


    def report(self):

        self.y_true_11 = self.y_true_11.reshape(-1)
        self.y_pred_11 = self.y_pred_11.reshape(-1)

        for score in range(1,20):
            score=score/20
            y_pre=self.y_pred_11>score

            tn, fp, fn, tp = confusion_matrix(self.y_true_11, y_pre).ravel()

            precision=precision_score(self.y_true_11,y_pre)
            recall=recall_score(self.y_true_11,y_pre)
            f1=f1_score(self.y_true_11,y_pre)


            print('for threshold: %.4f, tn: %d,fp: %d,fn: %d,tp: %d,precision: %.4f, '
                  'recall: %.4f, f1: %.4f'%(score, tn, fp, fn, tp,precision,recall,f1))


        return score




if __name__=='__main__':
    ROCAUC_score = ROCAUCMeter()

    y_true = np.random.randint(2, size=10000)
    y_prob = np.random.rand(10000)

    ROCAUC_score.update(y_true,y_prob)
    print(ROCAUC_score.avg)