
import sys
import torch
import numpy as np
import warnings
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score,average_precision_score, roc_curve,precision_recall_curve,accuracy_score,auc
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
        np.save('vgg_sleep_features_y_true_11.npy',self.y_true_11)
        np.save('vgg_sleep_features_y_pred_11.npy',self.y_pred_11)
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

    def report_with_recall_precision(self,):

        '''
        precisions, recalls, thresholds = precision_recall_curve(y_true_1, y_pre_1)
        youden = precisions + recalls
        cutoff = thresholds[np.argmax(youden)]
        y_pred_t = [1 if i > cutoff else 0 for i in y_pre_1]
        tn, fp, fn, tp = confusion_matrix(y_true_1, y_pred_t).ravel()
        recall = round(recall_score(y_true_1, y_pred_t), 4)
        precision = round(precision_score(y_true_1, y_pred_t), 4)
        f1 = round(f1_score(y_true_1, y_pred_t), 4)
        print('for recall: %.4f, precision: %.4f,tn: %d,fp: %d,fn: %d,tp: %d,f1: %.4f, cutoff: %.8f'
              % (recall, precision, tn, fp, fn, tp, f1, cutoff))
        '''
        self.y_true_11 = self.y_true_11.reshape(-1)
        self.y_pred_11 = self.y_pred_11.reshape(-1)
        y_true_1 = self.y_true_11
        y_pre_1 = self.y_pred_11
        precisions, recalls, thresholds = precision_recall_curve(y_true_1, y_pre_1)

        y_true_111 = y_true_1.reshape(-1)
        y_pred_111 = y_pre_1.reshape(-1)

        for i in range(1, 21):
            r = i / 20

            for j in range(len(recalls)):
                if float(recalls[j]) < float(r):
                    recall = round(recalls[j - 1], 9)
                    # threshold = round(thresholds[j - 1], 4)
                    threshold = thresholds[j - 1]
                    # print("threshold:", threshold)
                    # precision = round(precisions[j - 1], 4)
                    y_pre = y_pred_111 > threshold
                    tn, fp, fn, tp = confusion_matrix(y_true_111, y_pre).ravel()
                    precision = precision_score(y_true_111, y_pre)
                    recall = recall_score(y_true_111, y_pre)
                    f1 = f1_score(y_true_111, y_pre)
                    print('for recall: %.9f, tn: %d,fp: %d,fn: %d,tp: %d,precision: %.4f,f1: %.4f, threshold: %.8f' % (
                        recall, tn, fp, fn, tp, precision, f1, threshold))
                    break

    def report_tpr_acc(self, video_y_true, video_y_pre, base_y_true, base_y_pre, img_path):
        # 准确率
        # video_accuracy = accuracy_score(video_y_true, video_y_pre)
        base_accuracy = accuracy_score(self.y_true_11, self.y_pred_11)
        # 特异性
        # video_fpr, video_tpr, video_thresholds = roc_curve(video_y_true, video_y_pre)
        base_fpr, base_tpr, base_thresholds = roc_curve(self.y_true_11, self.y_pred_11)

        lines = []
        labels = []

        # l, = plt.plot(video_accuracy, video_tpr, color='navy', lw=2)  # 划线
        # lines.append(l)
        # labels.append('video_accuracy_sensitivity')

        l, = plt.plot(base_accuracy, base_tpr, color='darkorange', lw=2)  # 划线
        lines.append(l)
        labels.append('base_accuracy_sensitivity')

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0, 1.1])
        plt.xticks(np.linspace(0, 1.0, 10))
        plt.ylim([0, 1.1])
        plt.yticks(np.linspace(0, 1.0, 10))
        plt.xlabel('accuracy')
        plt.ylabel('sensitivity')
        plt.title(
            'accuracy sensitivity compare image')
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
        plt.savefig(img_path, dpi=300, bbox_inches='tight')

    def report_tpr_fpr(self, video_y_true, video_y_pre, base_y_true, base_y_pre, img_path):
        # 特异性
        # video_fpr, video_tpr, video_thresholds = roc_curve(video_y_true, video_y_pre)
        base_fpr, base_tpr, base_thresholds = roc_curve(self.y_true_11, self.y_pred_11)
        # video_auc2 = auc(video_fpr, video_tpr)
        base_auc2 = auc(base_fpr, base_tpr)
        # print('video  auc : {0:0.4f}'.format(
        #     video_auc2))
        print('base auc : {0:0.4f}'.format(
            base_auc2))

        lines = []
        labels = []

        fig11, ax = plt.subplots()
        # 将右侧和上方的边框线设置为不显示
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # l, = ax.plot(video_fpr, video_tpr, color='navy', lw=2)  # 划线
        # lines.append(l)
        # labels.append('video_specificiszty_sensitivity (auc = {0:0.4f})'
        #               ''.format(video_auc2))
        np.save('effv2-fpr.npy',base_fpr)
        np.save('effv2-tpr.npy',base_tpr)
        l, = ax.plot(base_fpr, base_tpr, color='darkorange', lw=2)  # 划线
        lines.append(l)
        labels.append('base_specificity_sensitivity (auc = {0:0.4f})'
                      ''.format(base_auc2))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)

        ax.set_xlim([0, 1.0])
        ax.set_xticks(np.linspace(0, 1.0, 11))
        ax.set_ylim([0, 1.0])
        ax.set_yticks(np.linspace(0, 1.0, 11))
        font = {'family': 'times new roman', 'size': 14}
        ax.set_xlabel('1-specificity', fontdict=font)
        ax.set_ylabel('sensitivity', fontdict=font)
        ax.set_title('specificity sensitivity compare image', fontdict=font)
        plt.legend(lines, labels, loc=(0, -.38), prop=font)
        plt.savefig(img_path, dpi=300, bbox_inches='tight')

    def report_with_recall(self, video_y_true, video_y_pre, base_y_true, base_y_pre, img_path):
        np.save('vgg16_pred.npy', self.y_pred_11)
        np.save('vgg16_true.npy', self.y_true_11)
        # video_precision, video_recall, video_thresholds = precision_recall_curve(video_y_true, video_y_pre)
        base_precision, base_recall, base_thresholds = precision_recall_curve(self.y_true_11, self.y_pred_11)
        # video_average_precision = average_precision_score(video_y_true, video_y_pre)
        base_average_precision = average_precision_score(self.y_true_11, self.y_pred_11)
        # print('video  precision-recall score: {0:0.4f}'.format(
        #     video_average_precision))
        print('base precision-recall score: {0:0.4f}'.format(
            base_average_precision))
        # print("precision:", precision, "recall:", recall, "thresholds:", thresholds)

        lines = []
        labels = []

        fig11, ax = plt.subplots()
        # 将右侧和上方的边框线设置为不显示
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # l, = plt.plot(video_recall, video_precision, color='navy', lw=2)
        # lines.append(l)
        # labels.append('add-video Precision-recall (area = {0:0.4f})'
        #               ''.format(video_average_precision))

        l, = plt.plot(base_recall, base_precision, color='darkorange', lw=2)
        np.save('effv2-recall.npy',base_recall)
        np.save('effv2-precision.npy',base_precision)
        lines.append(l)
        labels.append('base-line Precision-recall (area = {0:0.4f})'
                      ''.format(base_average_precision))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)

        ax.set_xlim([0, 1.0])
        ax.set_xticks(np.linspace(0, 1.0, 11))
        ax.set_ylim([0, 1.0])
        ax.set_yticks(np.linspace(0, 1.0, 11))
        font = {'family': 'times new roman', 'size': 14}
        ax.set_xlabel('Recall', fontdict=font)
        ax.set_ylabel('Precision', fontdict=font)
        ax.set_title('sprecision recall compare image', fontdict=font)
        plt.legend(lines, labels, loc=(0, -.38), prop=font)
        plt.savefig(img_path, dpi=300, bbox_inches='tight')


if __name__=='__main__':
    ROCAUC_score = ROCAUCMeter()

    y_true = np.random.randint(2, size=10000)
    y_prob = np.random.rand(10000)

    ROCAUC_score.update(y_true,y_prob)
    print(ROCAUC_score.avg)