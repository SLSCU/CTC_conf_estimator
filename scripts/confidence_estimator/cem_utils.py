import numpy as np
import torch
import multiprocessing as mp
import torch.nn.functional as F
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
import editdistance as ed

def rmse_wcr_score(y_true_list, y_pred_list):
    # y_true_list: (B, U)
    # y_pred_list: (B, U)
    utt_wcr = []
    utt_conf = []
    for bi in range(len(y_pred_list)):
        if len(y_pred_list[bi]) == 0:
            continue
        utt_wcr.append(sum(y_true_list[bi]) / len(y_pred_list[bi]))
        utt_conf.append(np.mean(y_pred_list[bi]))
    return mean_squared_error(utt_wcr, utt_conf, squared=False)

 
def rmse_wer_score(y_pred_list, text_predictions, text_groundtruths):
    # y_true_list: (B, U)
    # y_pred_list: (B, U)
    utt_wer = []
    utt_conf = []
    for bi in range(len(y_pred_list)):
        if len(y_pred_list[bi]) == 0 or len(text_groundtruths[bi]) == 0 or np.isnan(y_pred_list[bi]).any():
            continue
        word_error = ed.eval(text_predictions[bi].split(" "), text_groundtruths[bi].split(" ")) 
        utt_wer.append( word_error / len(text_groundtruths[bi].split(" ")))
        utt_conf.append(np.mean(y_pred_list[bi]))
        if np.isnan(np.array(utt_wer)).any():
            breakpoint()
        if np.isnan(utt_conf).any():
            breakpoint()
    return mean_squared_error( 1 - np.array(utt_wer), utt_conf, squared=False)


def calibrate_binning(y_true, y_pred, nbins=50):
    bins = np.linspace(0.0, 1.0, nbins + 1)
    # bin_ids = np.digitize(y_pred, bins, right=False)
    bin_ids = np.searchsorted(bins[1:-1], y_pred)

    bin_sums = np.bincount(bin_ids, weights=y_pred, minlength=len(bins))
    bin_true = np.bincount(bin_ids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(bin_ids, minlength=len(bins))

    # bin_total[bin_total == 0] = 1
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    return bin_ids, prob_true, prob_pred, bin_total[nonzero]

class NCE():
    def __init__(self):
        self.correct = 0.
        self.cross_ent = 0.
        self.n_samples = 0

    
    def update(self, correct_inc, conf, conf_len):
        with torch.no_grad():
            self.correct += correct_inc.sum().item()
            self.n_samples += conf_len.sum().item()
            mask = length_to_mask(conf_len, max_len=conf.shape[1], dtype=torch.float32)
            cross_ent = F.binary_cross_entropy(conf.squeeze(), correct_inc, reduction='none')
            self.cross_ent += (mask * cross_ent).sum().item()
                
    def get_nce(self):
        acc = self.correct / self.n_samples
        ent_ref = - acc * np.log(acc) - (1 - acc) * np.log(1-acc)
        ent_crs = self.cross_ent / self.n_samples
        return ( ent_ref - ent_crs ) / ent_ref

# https://www.isca-speech.org/archive_v0/archive_papers/eurospeech_1997/e97_0831.pdf
def norm_bicross_entropy(y_true, y_pred):
    acc = np.sum(y_true) / len(y_true)    
    ent_ref = - acc * np.log(acc) - (1 - acc) * np.log(1-acc)
    cross_ent = log_loss(y_true, y_pred)
    # print(ent_ref, cross_ent)
    return ( ent_ref - cross_ent ) / ent_ref



def get_alignment_label_thead(x_in, y_in):
    if torch.is_tensor(x_in):
        x_in = x_in.cpu().numpy()
    if torch.is_tensor(y_in):
        y_in = y_in.cpu().numpy()

    max_len = max([len(xi) for xi in x_in])
    pool_size = 8
    step_size = len(x_in) // pool_size + 1
    with mp.pool.ThreadPool(pool_size) as p:
        labs = p.map(get_alignment_label, [ (x_in[i:i+step_size], y_in[i:i+step_size], max_len) for i in range(0, len(x_in), step_size) ] )
    labels = np.concatenate(labs, axis=0)
    return labels
    

def get_alignment_label(params):
    input_string = True
    x_in, y_in, max_len = params
    # x : (batch, ids), y : ()
    if torch.is_tensor(x_in):
        x_in = x_in.cpu().numpy()
    if torch.is_tensor(y_in):
        y_in = y_in.cpu().numpy()

    labels = np.zeros( (len(x_in), max_len), dtype=np.float32) # (batch, maxlen)
    for bi, (xi, yi) in enumerate(zip(x_in, y_in)):
        x = list(xi)
        y = list(yi)
        Tx, Ty = len(x), len(y)

        dist = np.ones( (Tx+1, Ty+1), dtype=int ) * np.inf
        
        dist[0, 0] = 0
        dist[:, 0] = np.arange(Tx+1)
        dist[0, :] = np.arange(Ty+1)

        for i in range(1, Tx + 1):
            for j in range(1, Ty + 1):
                if x[i-1] == y[j-1]:      
                    dist[i, j] = min(dist[i-1, j-1], dist[i-1, j], dist[i, j-1])
                else:
                    dist[i, j] = min(dist[i-1, j-1], dist[i-1, j], dist[i, j-1]) + 1
        # print(dist)
        # backtrack
        aln = []
        i, j = Tx, Ty
        while (i > 0 and j > 0):            
            aln.append( (i,j) )
            dir = min(dist[i, j-1], dist[i-1,j-1], dist[i-1, j])
            if dir == dist[i-1, j-1]:                
                if dist[i][j] == dir:
                    labels[bi, i-1] = 1.
                i-=1; j-=1
            elif dir == dist[i, j-1]:
                j-=1
                # deletetion error here; we intentionally do not add label to
                # make the length of alignment and the length of prediction stay the same.                
            elif dir == dist[i-1, j]:
                i-=1            
            else: raise ValueError("{} {} {}".format(dist[i,j], i,j ))
    return labels


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask

def weighted_binary_cross_entropy(sigmoid_x, targets, pos_weight, weight=None, size_average=True, reduce=True):
    """
    Args:
        sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
        targets: true value, one-hot-like vector of size [N,C]
        pos_weight: Weight for postive sample
    """
    if not (targets.size() == sigmoid_x.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

    loss = -pos_weight* targets * sigmoid_x.log() - (1-targets)*(1-sigmoid_x).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


class WeightedBCELoss(torch.nn.Module):
    def __init__(self, pos_weight=1, weight=None, PosWeightIsDynamic= False, WeightIsDynamic= False, size_average=True, reduce=True):
        """
        Args:
            pos_weight = Weight for postive samples. Size [1,C]
            weight = Weight for Each class. Size [1,C]
            PosWeightIsDynamic: If True, the pos_weight is computed on each batch. If pos_weight is None, then it remains None.
            WeightIsDynamic: If True, the weight is computed on each batch. If weight is None, then it remains None.
        """
        super().__init__()

        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.size_average = size_average
        self.reduce = reduce
        self.PosWeightIsDynamic = PosWeightIsDynamic

    def forward(self, input, target):
        # pos_weight = Variable(self.pos_weight) if not isinstance(self.pos_weight, Variable) else self.pos_weight
        if self.PosWeightIsDynamic:
            positive_counts = target.sum(dim=0)
            nBatch = len(target)
            self.pos_weight = (nBatch - positive_counts)/(positive_counts +1e-5)

        if self.weight is not None:
            # weight = Variable(self.weight) if not isinstance(self.weight, Variable) else self.weight
            return weighted_binary_cross_entropy(input, target,
                                                 self.pos_weight,
                                                 weight=self.weight,
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)
        else:
            return weighted_binary_cross_entropy(input, target,
                                                 self.pos_weight,
                                                 weight=None,
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)

def compute_eer(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer

    
if __name__ == '__main__':
    # x_in = torch.IntTensor([ [1,2,3,4,0], [4,4,4,4,4], [3,3,4,4,0] ])
    # y_in = [ [1,2,3,4,0], [4,4,4,4], [3,4] ]
    # print(get_alignment_label_thead(x_in, y_in))
                
    # x_in = ["Join the dark forces haha".split(" "), "Join dark forces haha".split(" "), "Join dark forces hah ha".split(" ") ]
    # y_in = ["join the dunk forces".split(" ")] * 3 
    # print(get_alignment_label_thead(x_in, y_in))

    x_in = ["ซึ่ง ฝั่ง รัฐบาล นะ ครับ ก็ คง หวัง ว่า การ เป็นอิน์เน็ต ทำ ให้ โมตม อ การ ประท้วงนี่ ค่อย ค่อย ตายหาย ไป เอง นะ ครับ เพราะ ว่า คุย กัน ไม่ ได้ สื่อสร ม ได้ สัก พัก ก็ ง ลกเลกั ไ เง คง คิด อย่าง นี้".split(" ")]
    y_in = ["ซึ่ง ฝั่ง รัฐบาล ก็ คง หวัง ว่า การ ปิด อินเตอร์เน็ต จะ ทำ ให้ โมเมน ตั้ม ของ การ ประท้วง ค่อย ค่อย ตาย หาย ไป เอง เพราะ คุย กัน ไม่ ได้ สื่อสาร กัน ไม่ ได้ สัก พัก ก็ คง เลิก เลิก กัน ไป เอง คง คิด อย่าง นี้".split(" ")]
    #          1. 1. 1.    0. 0. 0. 1. 1.  1. 1.   0.      0. 1.
    # x_in = ["ซึ่ง ฝั่ง รัฐบาล นะ ครับ ก็ คง หวัง ว่า การ เป็นอิน์เน็ต ทำ ให้".split(" ")]
    # y_in = ["ซึ่ง ฝั่ง รัฐบาล ก็ คง หวัง ว่า การ ปิด อินเตอร์เน็ต จะ ทำ ให้".split(" ")]
    # import sys
    # np.set_printoptions(threshold=sys.maxsize)
    print(get_alignment_label( (x_in, y_in, len(x_in[0])) ))



