import cgi
import pickle
from pickletools import optimize
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from cem_utils import length_to_mask, get_alignment_label_thead, WeightedBCELoss
import pwlf
from tqdm import tqdm
from cem_utils import norm_bicross_entropy, rmse_wcr_score, rmse_wer_score, calibrate_binning
import sklearn.metrics as metrics 
import scipy as spy

class CEM(nn.Module):

    def __init__(self, dim_in, dim_out, blank_index, space_index, 
            ctc_agg='max', word_agg='mean', use_logits=True, pos_weight=1,
            is_add_onehot=False, is_add_counter=False, is_add_variance=False, is_add_logsoftmax=False,
            is_add_selfattn=False, is_add_full_selfattn=False, inplace_selfattn=False, topk=-1, attn_dim=256,
            split_token = ' ', is_add_transformer=False, n_fc_layer = 1, linear_activation=None,
            use_context_heads=0, cnn_kernel_size=0, is_add_softmax=False, temp=1,
            is_boundary_inclusive=False, skip_blank=True,
        ):
        super().__init__()
        
        self.default_blank_prob = 0.9999
        self.blank_index = blank_index
        self.space_index = space_index
        self.dim_in = dim_in
        self.n_fc_layer = n_fc_layer

        self.ctc_agg = ctc_agg # min, max      
        self.word_agg = word_agg 
        self.split_token = split_token
        self.use_logits = use_logits
        self.criteria = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight), reduction='none')
        self.use_context_heads = use_context_heads

        self.is_add_onehot = is_add_onehot # use concat(asr_features, onehot) for cem
        self.is_add_counter = is_add_counter
        self.is_add_variance = is_add_variance
        self.is_add_logsoftmax = is_add_logsoftmax
        self.is_add_softmax = is_add_softmax
        self.is_add_selfattn = is_add_selfattn
        self.topk = topk
        self.is_add_full_selfattn = is_add_full_selfattn
        self.inplace_selfattn = inplace_selfattn
        self.is_add_transformer = is_add_transformer
        self.cnn_kernel_size = cnn_kernel_size
	    
        if self.topk > 0:    
            in_features= topk
        else:
            in_features = dim_in

        if is_add_onehot:
            in_features += dim_in
        if is_add_counter:
            in_features += 1
        if is_add_variance:
            in_features += dim_in
        if is_add_logsoftmax:    
            in_features += self.topk if self.topk > 0 else dim_in
        if is_add_softmax:
            in_features += self.topk if self.topk > 0 else dim_in
        if use_context_heads > 0:
            in_features *= (use_context_heads + 1) # 

        if is_add_selfattn:
            in_features += attn_dim
            if inplace_selfattn:
                in_features -= dim_in            
            self.projection = nn.Linear(in_features=dim_in, out_features=attn_dim, bias=True)
            self.self_attn = nn.MultiheadAttention(attn_dim, 4, dropout=0.1)
            self.fc = nn.Linear(in_features=in_features, out_features=1, bias=True) 
        elif is_add_full_selfattn:
            self.projection = nn.Linear(in_features=in_features, out_features=attn_dim, bias=True)
            self.self_attn = nn.MultiheadAttention(attn_dim, 4, dropout=0.1)
            self.fc = nn.Linear(in_features=attn_dim, out_features=1, bias=True)
        elif is_add_transformer:
            self.projection = nn.Linear(in_features=in_features, out_features=attn_dim, bias=True)
            encoder_layer = nn.TransformerEncoderLayer(d_model=attn_dim, nhead=1, dropout=0.1)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.fc = nn.Linear(in_features=attn_dim, out_features=1, bias=True)
        else:   
            if self.cnn_kernel_size > 0:
                self.cnn = nn.Conv1d(in_features, 256, self.cnn_kernel_size , stride=1, padding='same')
                in_features = 256

            if n_fc_layer == 1:
                self.fc = nn.Linear(in_features=in_features, out_features=1, bias=True) 
            elif linear_activation == None:
                self.fc = nn.Sequential(
                    nn.Linear(in_features, 256),
                    nn.Linear(256, 128),
                    nn.Linear(128, 1),
                    )
            elif linear_activation == 'relu':
                self.fc = nn.Sequential(
                    nn.Linear(in_features, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    )
            elif linear_activation == 'swish': 
                self.fc = nn.Sequential(
                    nn.Linear(in_features, 256),
                    nn.SiLU(),
                    nn.Linear(256, 128),
                    nn.SiLU(),
                    nn.Linear(128, 1),
                    )
            else:
                raise NotImplementedError(linear_activation)
        self.temp = temp
        self.is_boundary_inclusive = is_boundary_inclusive
        self.skip_blank = skip_blank
        assert not (is_add_selfattn & is_add_full_selfattn)
        

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


    def forward(self, features, mask=None, use_logits=False):
        # features: (B, T, C)
        if self.cnn_kernel_size > 0:
            features = self.cnn(features.permute(0, 2, 1))
            features = features.permute(0, 2, 1)

        if self.is_add_selfattn:
            x_proj = self.projection(features[:,:,:self.dim_in].permute(1,0,2)) # (seq, batch, feature).
            # (T, B, C)
            attn_output, attn_output_weights = self.self_attn(x_proj, x_proj, x_proj, key_padding_mask=~mask)
            # (B, T, dim_in)
            attn_output = attn_output.permute(1,0,2)
            if self.inplace_selfattn:
                features = torch.cat([features[:,:,self.dim_in:], attn_output], dim=-1)
            else:
                features = torch.cat([features, attn_output], dim=-1)
        elif self.is_add_full_selfattn:
            x_proj = self.projection(features.permute(1,0,2)) # (seq, batch, feature). 
            attn_output, attn_output_weights = self.self_attn(x_proj, x_proj, x_proj, key_padding_mask=~mask)
            features = attn_output.permute(1,0,2)
        
        elif self.is_add_transformer:
            x_proj = self.projection(features.permute(1,0,2)) # (seq, batch, feature).  
            attn_output = self.transformer(x_proj, src_key_padding_mask=~mask)
            features = attn_output.permute(1,0,2)

        x = self.fc(features) # (B, T)
        if use_logits:
            return x
        else:
            return torch.sigmoid(x) # (B, T)


    def training_step(self, asr_model, batch, batch_idx):    
        # generate training data
        # (B, U, C), (B, U), (B, )
        features, conf_gt, features_len = self.get_features(asr_model, batch, temp=self.temp)
        masking = length_to_mask(features_len, max_len=features.shape[1], dtype=torch.torch.bool).cuda() # (B, U)
        features, conf_gt = features.cuda(), conf_gt.cuda()
        # estimate confidence
        # logits
        x = self.forward(features, masking, use_logits=True).squeeze()        

        loss = self.criteria(x, conf_gt)
        loss[~masking] = 0
        return loss.sum() / features_len.sum()
    

    def inference(self, asr_model, batch, skip_cem=False):
        # generate training data
        self.eval()
        with torch.no_grad():
            # features is word level logits
            if skip_cem:
                features, conf_gt, features_len, text_predictions, text_references = self.get_features(
                    asr_model, batch, full_return=True, temp=self.temp, skip_cem=True)
                x = features

            # estimate confidence
            else:
                features, conf_gt, features_len, text_predictions, text_references = self.get_features(
                    asr_model, batch, full_return=True, temp=self.temp)
                masking = length_to_mask(features_len, max_len=features.shape[1], dtype=torch.torch.bool).cuda()
                x = self.forward(features.cuda(), masking, use_logits=False).squeeze().cpu()
                if len(x.shape) == 1:
                    x = x.unsqueeze(dim=0)

            # formating outputs
            conf = []
            conf_gt_flatten = []
            for bi in range( x.shape[0] ):
                try:
                    conf.append(x[bi, :features_len[bi]].tolist())
                    conf_gt_flatten.append( conf_gt[bi, :features_len[bi]].tolist() )
                except Exception as  E:
                    print(conf_gt)
                    print(features_len)
                    print(E)
        self.train()
        return conf, conf_gt_flatten, text_predictions, text_references


    def configure_optimizers(self,):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        

    def get_features(self, asr_model, batch, full_return=False, temp=1., skip_cem=False):
        signal, signal_len, transcript, transcript_len = batch
        if torch.cuda.is_available():
            signal, signal_len, transcript, transcript_len = signal.cuda(), signal_len.cuda(), transcript.cuda(), transcript_len.cuda()
        # generate training data
        with torch.no_grad():
            # ASR models always return logits because merging consucutive duplicates doesn't make sense for logprob
            asr_outputs = asr_model.forward(input_signal=signal, input_signal_length=signal_len, use_logits=False)
            if len(asr_outputs) == 5:
                logits_mid, logits_left, logits_right, encoded_len, greedy_predictions = asr_outputs
            else:
                logits_mid, encoded_len, greedy_predictions = asr_outputs
            
            logits_mid = logits_mid/temp

            text_references = []

            for batch_ind in range(greedy_predictions.shape[0]):
                seq_len = transcript_len[batch_ind].cpu().detach().numpy()
                seq_ids = transcript[batch_ind].cpu().detach().numpy()
                reference = ''.join(asr_model._wer.decoding.decode_tokens_to_str(seq_ids[0:seq_len]))
                text_references.append(reference) 
            # (B, T) , (B, T, C), (B, )
            if self.use_context_heads > 0:
                pred, pred_prob, pred_prob_left, pred_prob_right, pred_len = self.merge_repeated_batch_with_context(
                    logits_mid, logits_left, logits_right, skip_blank=self.skip_blank, agg=self.ctc_agg
                )
            else:
                pred, pred_prob, pred_len = self.merge_repeated_batch(logits_mid, skip_blank=self.skip_blank, agg=self.ctc_agg)
            text_predictions = [ asr_model._wer.decoding.decode_tokens_to_str(pred[i, :pred_len[i]]) 
                for i in range(len(greedy_predictions)) ]
            conf_gt = self.generate_labels(text_predictions, text_references, split_token=self.split_token)            
            # tensor (B, T, C1)
            if self.split_token == None:
                word_agg = None
            else: word_agg = self.word_agg

            if skip_cem:
                aggregation = self.aggregation_probs
            else:
                aggregation = self.aggregation_logits

            features, features_len = aggregation(pred, pred_prob, pred_len, maxlen=conf_gt.shape[1], by=self.space_index, agg=word_agg)
            features_len = pred_len if word_agg == None else features_len
            if self.use_context_heads > 0:
                features_left, _  = aggregation(pred, pred_prob_left, pred_len, maxlen=conf_gt.shape[1], by=self.space_index, agg=word_agg)
                features_right, _ = aggregation(pred, pred_prob_right, pred_len, maxlen=conf_gt.shape[1], by=self.space_index, agg=word_agg)
                # (B, T, 3*C1)
                features = torch.cat([features, features_left, features_right], dim=-1)

        if not full_return:
            return features, conf_gt, features_len
        else:
            return features, conf_gt, features_len, text_predictions, text_references


    def generate_labels(self, predictions, groundtruths, split_token=None):
        if split_token != None:
            w_pred = [ pred.split(split_token) for pred in predictions ]
            w_gt = [ gt.split(split_token) for gt in groundtruths ]
            return torch.tensor(get_alignment_label_thead(w_pred, w_gt))
        else:
            return torch.tensor(get_alignment_label_thead( list(predictions), list(groundtruths) ))


    def aggregation_probs(self, pred, pred_softmaxs, pred_len, maxlen, by=0, agg='mean'):
        # pred;  (B, T)
        # pred_logits; (B, T, C)
        # maxlen : int ; the biggest number of word across samples inside the batch
        # by; int := blank_index ; index used for spliting
        # agg; mean, last, max    
        # return features (B, maxlen, C)
        
        B, T, C = pred_softmaxs.shape

        if torch.is_tensor(pred_softmaxs):
            # pred_softmaxs = pred_logits.softmax()
            pred_softmaxs = pred_softmaxs.cpu().numpy()
        # else:
        #     pred_softmaxs = spy.special.softmax(pred_logits, axis=-1)

        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()

        cur_position = np.full((B,), 0, dtype=np.int64) # indicates position in aggregated_prob
        token_counter = np.full( (B), 0, dtype=np.float32) # for mean

        default_prob = 0.
        
        candidate = np.full( (B,), default_prob, dtype=np.float32)
        aggregated_probs = np.full( (B, maxlen), 0., dtype=np.float32)

        x = pred_softmaxs.argmax(axis=2) # batch_size, output_len
        v = pred_softmaxs.max(axis=2) # batch_size, output_len -> prob
        have_found_character = np.full( (B,), False, dtype=np.bool)

        if agg != None:
            for t in range(0, pred.shape[1]):
                found_boundary = (pred[:, t] == by) & (t < pred_len)

                # update states
                if agg in ['mean', 'geomean']:
                    candidate[found_boundary] /= token_counter[found_boundary]
                    candidate[~found_boundary] += v[~found_boundary, t]
                    # print(candidate[found_boundary])
                elif agg == 'last':
                    candidate[~found_boundary] = v[~found_boundary, t]
                elif agg == 'min':
                    update = candidate > v[:, t]
                    # candidate is the up to date lowest logits within the current word boundary
                    candidate[~found_boundary & update] = v[~found_boundary & update, t]
                else:
                    raise NotImplementedError("agg:={} have not implemented.".format(agg))

                # update results
                aggregated_probs[have_found_character, cur_position[have_found_character]] = candidate[have_found_character]
                
                # reset states
                candidate[found_boundary] = default_prob
                if self.is_boundary_inclusive:
                    candidate[found_boundary] = v[found_boundary, t]
                    token_counter[found_boundary] = 1.
                else:
                    candidate[found_boundary] = default_prob
                    token_counter[found_boundary] = 0.
                
                token_counter[~found_boundary] += 1.
                # update iterator
                cur_position[found_boundary & have_found_character] += 1
                have_found_character = ~found_boundary
 
            # Finalized the lastest alphabets
            if agg in ['mean', 'geomean']:
                candidate[~found_boundary] /= token_counter[~found_boundary]
                aggregated_probs[np.arange(B), cur_position] = candidate
            
            if agg == 'geomean':
                # convert back to probability scales
                aggregated_probs = torch.tensor(np.exp(aggregated_probs))

            if agg == 'min':
                aggregated_probs[np.arange(B), cur_position] = candidate
                aggregated_probs[np.isinf(aggregated_probs)] = 0.                

        else:
            aggregated_probs = pred_softmaxs

        aggregated_probs = torch.tensor(aggregated_probs)
        agg_len = torch.tensor(cur_position + 1) # surplus the last word into length
        return aggregated_probs, agg_len
    

    def aggregation_logits(self, pred, pred_logits, pred_len, maxlen, by=0, agg='mean'):
        # pred;  (B, T)
        # pred_logits; (B, T, C)
        # maxlen : int ; the biggest number of word across samples inside the batch
        # by; int := blank_index ; index used for spliting
        # agg; mean, last, max    
        # return features (B, maxlen, C)
        
        assert agg in ['min', 'mean', 'last', 'sum']
        
        B, T, C = pred_logits.shape
        if torch.is_tensor(pred_logits):
            pred_logits = pred_logits.cpu().numpy()
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()

        cur_position = np.full((B,), 0, dtype=np.int64) # indicates position in aggregated_logits
        token_counter = np.full( (B, 1), 0, dtype=np.float32) # for mean
        have_found_character = np.full( (B,), False, dtype=np.bool)

        if self.is_add_counter:
            aggregated_sizes = np.full( (B, maxlen, 1), 0., dtype=np.float32)

        if self.is_add_onehot:
            candidate_onehot = np.full( (B, C), 0., dtype=np.float32)
            aggregated_onehots = np.full( (B, maxlen, C), 0., dtype=np.float32)

        if self.is_add_variance:
            # TODO: implement this
            pass

        if self.topk > 0:
            C = self.topk
            pred_logits, topk_index = torch.tensor(pred_logits).topk(self.topk)
            pred_logits = pred_logits.numpy()

        if self.use_logits: # logit mode
            if agg == 'geomean':
                default_prob = np.full( (C, ), 1., dtype=np.float32)
            elif agg == 'min':
                default_prob = np.full( (C, ), np.inf, dtype=np.float32)
            else:
                default_prob = np.full( (C, ), 0., dtype=np.float32)
        else: # logprob mode
            # raise NotImplementedError()
            default_prob = np.full( (C, ), np.log((1-self.default_blank_prob)/(C-1)), dtype=np.float32)
            # default_prob[self.blank_index] = np.log(self.default_blank_prob)
        
        candidate = np.full( (B, C), default_prob, dtype=np.float32)
        aggregated_logits = np.full( (B, maxlen, C), 0., dtype=np.float32)

        if agg == 'min':
            x = pred_logits.argmax(axis=2) # batch_size, output_len
            v = pred_logits.max(axis=2) # batch_size, output_len

        if agg == 'geomean':
            # turn into log
            pred_logits = np.log(pred_logits)

        if agg != None:
            for t in range(0, pred.shape[1]):
                found_boundary = (pred[:, t] == by) & (t < pred_len)
                # print(candidate)
                if ( np.isinf(candidate)).any():
                    breakpoint()
                # update states
                if agg in ['mean', 'geomean']:
                    candidate[found_boundary, :] /= token_counter[found_boundary]
                    candidate[~found_boundary] += pred_logits[~found_boundary, t, :]
                elif agg == 'sum':
                    candidate[~found_boundary] += pred_logits[~found_boundary, t, :]
                elif agg == 'last':
                    candidate[~found_boundary] = pred_logits[~found_boundary, t, :]
                elif agg == 'min':
                    update = candidate.max(axis=-1) > v[:, t]
                    # candidate is the up to date lowest logits within the current word boundary
                    candidate[~found_boundary & update, :] = pred_logits[~found_boundary&update, t, :]
                else:
                    raise NotImplementedError("agg:={} have not implemented.".format(agg))

                # update results
                aggregated_logits[have_found_character, cur_position[have_found_character]] = candidate[have_found_character]
                if self.is_add_counter:
                    aggregated_sizes[found_boundary, cur_position[found_boundary]] = token_counter[found_boundary]
                if self.is_add_onehot:
                    aggregated_onehots[found_boundary, cur_position[found_boundary]] = candidate_onehot[found_boundary]

                # reset states
                if self.is_boundary_inclusive:
                    candidate[found_boundary] = pred_logits[found_boundary, t, :]
                    token_counter[found_boundary] = 1.
                else:
                    candidate[found_boundary] = default_prob
                    token_counter[found_boundary] = 0.
                if self.is_add_onehot:
                    candidate_onehot[~found_boundary, pred[~found_boundary, t]] += 1.
                    candidate_onehot[found_boundary, pred[found_boundary, t]] = 0.

                # update iterator
                cur_position[found_boundary & have_found_character] += 1
                have_found_character = ~found_boundary
                token_counter[~found_boundary] += 1.
 
            # Finalized the lastest alphabets
            if agg in ['mean', 'geomean']:
                candidate[~found_boundary, :] /= token_counter[~found_boundary]
                aggregated_logits[np.arange(B), cur_position] = candidate
            
            if agg == 'sum':
                aggregated_logits[np.arange(B), cur_position] = candidate

            if agg == 'geomean':
                # convert back to probability scales
                aggregated_logits = torch.tensor(np.exp(aggregated_logits))

            if agg == 'min':
                aggregated_logits[np.arange(B), cur_position] = candidate
                aggregated_logits[np.isinf(aggregated_logits)] = 0.

            if self.is_add_counter:
                aggregated_sizes[~found_boundary, cur_position[~found_boundary]] = token_counter[~found_boundary]
                aggregated_sizes = torch.tensor(aggregated_sizes) 
            
            if self.is_add_onehot:
                aggregated_onehots[~found_boundary, cur_position[~found_boundary]] = candidate_onehot[~found_boundary]
                aggregated_onehots = torch.tensor(aggregated_onehots)
        else:
            aggregated_logits = pred_logits
            aggregated_onehots = F.one_hot(pred_logits.argmax(dim=-1))
	

        aggregated_logits = torch.tensor(aggregated_logits)
        agg_len = torch.tensor(cur_position + 1) # surplus the last word into length
    
        features = aggregated_logits
        if self.is_add_onehot:
            features = torch.cat([features, aggregated_onehots], dim=-1) #(B, maxlen, 2C)
        if self.is_add_counter:
            features = torch.cat([features, aggregated_sizes], dim=-1) #(B, maxlen, 2C)
        if self.is_add_logsoftmax:
            features = torch.cat([features, F.log_softmax(aggregated_logits, dim=-1)], dim=-1)
        if self.is_add_softmax:
            features = torch.cat([features, F.softmax(aggregated_logits, dim=-1)], dim=-1)

        # (B, maxlen, dim), (B, )
        return features, agg_len
    

    def merge_repeated_batch(self, logits, blank_index=None, skip_blank=False, agg='max'):
        # x : (B, T)
        # input logits : batch_size, output_len, num_classes
        # return h : index of logits, which is has the highest probs for merged alphabets
        
        if(torch.is_tensor(logits)):   
            logits = logits.cpu().numpy()
        if blank_index == None: blank_index = self.blank_index

        B, T, C =logits.shape
        x = logits.argmax(axis=2) # batch_size, output_len
        v = logits.max(axis=2) # batch_size, output_len

        # TODO store index in h_prob for efficiency
        # h_prob = np.full( (B, T), 0, dtype=np.int64) # keep index of logits 
        if self.use_logits: # logit mode
            default_prob = np.full( (C, ), 0., dtype=np.float32)
        else: # logprob mode
            default_prob = np.full( (C, ), np.log((1-self.default_blank_prob)/(C-1)), dtype=np.float32)
            default_prob[self.blank_index] = np.log(self.default_blank_prob)
        h_prob = np.full( (B, T, C), default_prob, dtype=np.float32) # target logprobs

        h = np.full( (B, T), 0, dtype=np.int64) # alphabet list
        cur_position = np.full((B,), -1, dtype=np.int64) # position on h

        if agg == 'mean':
            default_candidate = np.full( (B, C), default_prob, dtype=np.float32)
            candidate = np.copy(default_candidate)
            frame_counter = np.zeros( (B,1), dtype=np.float32 )

        # h[:, 0] = x[:, 0].copy()
        cur_max_v = np.full( (B, ), 0, dtype=np.int32) # position of the logits for the latest alphabet        
        if not skip_blank:
            not_blank = x[:, 0] == x[:, 0]
        
        for t in range(0, h.shape[1]):
            if skip_blank: not_blank = x[:,t] != blank_index
            if t > 0: change = (x[:, t] != x[:, t-1])
            else: change = not_blank

            if agg == 'mean':
                candidate[change & not_blank] = logits[change & not_blank, t]
                candidate[~change & not_blank] += logits[~change & not_blank, t]
                frame_counter[change & not_blank] = 1
                frame_counter[~change & not_blank] += 1
                
                # reset state for blank candidates
                candidate[~not_blank] = np.full( (np.sum(~not_blank), C), default_prob, dtype=np.float32)
                frame_counter[~not_blank] = 0

                cur_position[change & not_blank] += 1
                selected_frames = (cur_position >= 0) & (frame_counter > 0).flatten()

                h_prob[selected_frames, cur_position[selected_frames]] = candidate[selected_frames] / frame_counter[selected_frames]
                h[selected_frames, cur_position[selected_frames]] = x[selected_frames, t]
 
            elif agg in ['max', 'min']:
                # Use max prob for for merged postition
                if agg == 'max':
                    update = (v[np.arange(B), cur_max_v] < v[:, t]) & not_blank # (B, T)
                elif agg == 'min':
                    update = (v[np.arange(B), cur_max_v] > v[:, t]) & not_blank # (B, T)
                cur_max_v[ change | update ] = t

                cur_position[change & not_blank] += 1
                selected_frames = (cur_position >= 0) & not_blank

                h_prob[selected_frames, cur_position[selected_frames]] = logits[selected_frames, cur_max_v[selected_frames]]
                h[selected_frames, cur_position[selected_frames]] = x[selected_frames, t]
            else:
                raise ValueError(f"agg {agg} is not implemented")
        
        h_len = cur_position + 1 # plus one converting indices to lengths
        # (B, T) , (B, T, C), (B, )
        return h, h_prob, h_len


    def merge_repeated_batch_with_context(self, logits, logits_left, logits_right, blank_index=None, skip_blank=False, 
        agg='max', agg_context='mean'):
        # x : (B, T)
        # input logits : batch_size, output_len, num_classes
        # input logits_left, logits_right: (nhead, batch_size, output_len, num_classes)
        # return h : index of logits, which is has the highest probs for merged alphabets
        
        if(torch.is_tensor(logits)):   
            logits = logits.cpu().numpy()
        if blank_index == None: blank_index = self.blank_index

        B, T, C =logits.shape
        x = logits.argmax(axis=2) # batch_size, output_len
        v = logits.max(axis=2) # batch_size, output_len

        # TODO store index in h_prob for efficiency
        # h_prob = np.full( (B, T), 0, dtype=np.int64) # keep index of logits 
        if self.use_logits: # logit mode
            default_prob = np.full( (C, ), 0., dtype=np.float32)
        else: # logprob mode
            default_prob = np.full( (C, ), np.log((1-self.default_blank_prob)/(C-1)), dtype=np.float32)
            default_prob[self.blank_index] = np.log(self.default_blank_prob)
        h_prob = np.full( (B, T, C), default_prob, dtype=np.float32) # target logprobs

        h = np.full( (B, T), 0, dtype=np.int64) # alphabet list
        cur_position = np.full((B,), 0, dtype=np.int64) # position on h

        # init context probs
        if agg_context == 'mean':
            logits_left = torch.stack(logits_left, dim=0).mean(dim=0).cpu().numpy() # (B, T, C)
            logits_right = torch.stack(logits_right, dim=0).mean(dim=0).cpu().numpy() # (B, T, C)
        else:
            raise ValueError(f"agg_context: {agg_context}")
        h_prob_left = np.full( (B, T, C), default_prob, dtype=np.float32) # context logprobs
        h_prob_right = np.full( (B, T, C), default_prob, dtype=np.float32) # context logprobs

        # h[:, 0] = x[:, 0].copy()
        cur_max_v = np.full( (B, ), 0, dtype=np.int32) # position of the logits for the latest alphabet        
        if not skip_blank:
            not_blank = x[:, 0] == x[:, 0] 
        
        for t in range(0, h.shape[1]):
            if skip_blank: not_blank = x[:,t] != blank_index
            if t > 0: change = (x[:, t] != x[:, t-1]) & not_blank
            else: change = not_blank

            # Use max prob for for merged postition
            if agg == 'max':
                update = (v[np.arange(B), cur_max_v] < v[:, t]) & not_blank # (B, T)
            elif agg == 'min':
                update = (v[np.arange(B), cur_max_v] > v[:, t]) & not_blank # (B, T)
            else:
                raise ValueError("agg {} is not implemented")
            cur_max_v[ change | update ] = t
            
            h_prob[np.arange(B), cur_position] = logits[np.arange(B), cur_max_v]
            h_prob_left[np.arange(B), cur_position] = logits_left[np.arange(B), cur_max_v]
            h_prob_right[np.arange(B), cur_position] = logits_right[np.arange(B), cur_max_v]
            h[np.arange(B), cur_position] = x[:, t]
            # print(change.shape, cur_position.shape, v.shape)
            # print( h_prob[change, cur_position[change]].shape)
            # print( h_prob[change, cur_position])
            # h_prob[change, cur_position[change], :] = logits[change, cur_max_v[change], :]
            # h[ change, cur_position[change]] = x[change, t]
            cur_position[change] += 1
        
        h_len = cur_position
        # (B, T) , (B, T, C), (B, )
        return h, h_prob, h_prob_left, h_prob_right, h_len


    def __str__(self):
        return f"""
        self.ctc_agg : {self.ctc_agg }
        self.word_agg: {self.word_agg}
        self.n_fc_layer: {self.n_fc_layer}
        self.split_token: {self.split_token}
        self.use_logits: {self.use_logits}
        self.is_add_onehot: {self.is_add_onehot}
        self.is_add_counter : {self.is_add_counter }
        self.is_add_variance : {self.is_add_variance }
        self.is_add_logsoftmax : {self.is_add_logsoftmax }
        self.is_add_selfattn : {self.is_add_selfattn }
        self.is_add_full_selfattn : {self.is_add_full_selfattn }
        self.topk : {self.topk }
        self.inplace_selfattn: {self.inplace_selfattn}
        self.is_add_transformer: {self.is_add_transformer}
        self.use_context_heads: {self.use_context_heads}
        self.cnn_kernel_size: {self.cnn_kernel_size}
        self.is_add_softmax: {self.is_add_softmax}
        """

class CEM_PWLF(CEM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pwlf = None

    def fit_pwlf(self, asr_model, dataloader, n_segments=5, limit_batch=-1, nbins=50):
        y_score, y_true = [], []
        with torch.no_grad():
            size = len(dataloader)
            asr_model.eval()
            self.eval()
            for batch_idx, batch in tqdm(enumerate(dataloader), total=size):
                # list of list (B, T)
                conf, conf_gt_flatten, text_predictions, text_references = super().inference(asr_model, batch)
                for bi in range(len(conf)):
                    y_score += conf[bi]
                    y_true += conf_gt_flatten[bi]
                # nce.update(w_gt, pred_conf, feats_len) 
                if limit_batch > 0 and batch_idx + 1 >= limit_batch:
                    break
            self.train()
            asr_model.train()
        y_score = np.array(y_score)
        y_true = np.array(y_true)

        if nbins > 0:
            bin_ids, prob_true, prob_pred, bin_total = calibrate_binning(y_true, y_score, nbins=nbins)
            y_true_binned = np.take(prob_true, bin_ids, axis=None)
            nce = norm_bicross_entropy(y_true, y_score)
            auc = metrics.roc_auc_score(y_true, y_score, average='macro')
            wcr = np.sum(np.abs(prob_true - prob_pred)) / len(prob_pred)
            my_pwlf = pwlf.PiecewiseLinFit(y_score, y_true_binned)
        else:
            nce = norm_bicross_entropy(y_true, y_score)
            auc = metrics.roc_auc_score(y_true, y_score, average='macro')
            wcr = np.sum(np.abs(y_true - y_score)) / len(y_score)
            my_pwlf = pwlf.PiecewiseLinFit(y_score, y_true)

        res = my_pwlf.fit(n_segments)
        y_score_pwlf = my_pwlf.predict(y_score)
        _, prob_true_pwlf, prob_pred_pwlf, bin_total = calibrate_binning(y_true, y_score_pwlf)
        
        nce_pwlf = norm_bicross_entropy(y_true, y_score_pwlf)
        auc_pwlf = metrics.roc_auc_score(y_true, y_score_pwlf, average='macro')
        wcr_pwlf = np.sum(np.abs(prob_true_pwlf - prob_pred_pwlf)) / len(prob_pred_pwlf) 
        print(f"nce: {nce} nce_pwlf: {nce_pwlf}")
        print(f"auc: {auc} auc_pwlf: {auc_pwlf}")
        print(f"mae_wcr: {wcr} mae_wcr_pwlf: {wcr_pwlf}")

        self.pwlf = my_pwlf

    
    def inference(self, *args, **kwargs):
        conf, conf_gt_flatten, text_predictions, text_references = super().inference( *args, **kwargs)
        if self.pwlf is not None:
            conf = [self.pwlf.predict(conf_b) for conf_b in conf]
            conf = [np.maximum(np.minimum(1., conf_b), 0.) for conf_b in conf]
        return conf, conf_gt_flatten, text_predictions, text_references 


if __name__ == '__main__':

    cem = CEM(10,10,0,3)
    # x_in = torch.IntTensor([ [1,2,3,4,0], [4,4,4,4,4], [3,3,4,4,0] ])
    # y_in = [ [1,2,3,4,0], [4,4,4,4], [3,4] ]
    x_in = ["Join the dark forces haha".split(" "), "Join dark forces haha".split(" "), "Join dark forces hah ha".split(" ") ]
    y_in = ["join the dunk forces".split(" ")] * 3 
    print(cem.get_alignment_label(x_in, y_in))
                


    # pred = torch.IntTensor([ [1,2,0,4], [4,0,4,4,4], [3,3,0,4] ])
    pred = torch.IntTensor([ [1,2,0,2] , [2,0,1,1]] )
    pred_prob = torch.tensor([
        np.log(np.array([
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ])),
       np.log(np.array([
           [0.1, 0.1, 0.7, 0.1],
           [0.7, 0.1, 0.1, 0.1],
           [0.1, 0.7, 0.1, 0.1],
           [0.1, 0.7, 0.1, 0.1],
       ])),
    #    np.log(np.array([
    #        [0.1, 0.7, 0.1, 0.1],
    #        [0.1, 0.1, 0.7, 0.1],
    #        [0.7, 0.1, 0.1, 0.1],
    #        [0.1, 0.1, 0.1, 0.7],
    #    ])),
    ]) # B,T,C
    print(pred)
    print(pred_prob.shape)
    print(cem.aggregation(pred, pred_prob, maxlen=2, by=0, agg='mean'))
