import pickle
import re
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from argparse import ArgumentParser
from CEM_module import CEM, CEM_PWLF
import logging
from nemo.collections.asr.models import EncDecCTCModel, EncDecCCTCModel, EncDecCTCModelBPE, EncDecRNNTBPEModel, EncDecRNNTModel
from nemo.collections.asr.modules import SpectrogramAugmentation

import torch
from tqdm import tqdm
from omegaconf import open_dict
import numpy as np
import torch.nn.functional as F
import datetime
from cem_utils import norm_bicross_entropy, rmse_wcr_score, rmse_wer_score, calibrate_binning
import sklearn.metrics as metrics # import roc_auc_score, precision_recall_curve, PrecisionRecallDisplay
from sklearn.calibration import calibration_curve, CalibrationDisplay
import editdistance as ed

import glob
import matplotlib.pyplot as plt
import re

plt.rcParams.update({'font.size': 16})
plt_figsize=(8, 4)

def get_parser():
    parser = ArgumentParser()
    # parser.add_argument(
    #     "--asr_model", type=str, default="QuartzNet15x5Base-En", required=True, help="Pass: 'QuartzNet15x5Base-En'",
    # )
    # parser.add_argument("--dataset", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--cctc', action='store_true', help='uses CCTC models')
    parser.add_argument('--rnnt', action='store_true', help='uses RNNT models')
    parser.add_argument('--bpe', action='store_true', help='use bpe models')
    parser.add_argument('--preds_dir', default='preds',  help='A directory sotring prediction results.')
    # parser.add_argument('--sr', default=8000,  help='Sampling rate.', type=int) 
    parser.add_argument('--lang', default='th',  help='language')  
    parser.add_argument('--retokenize', action='store_true',  help='retokenize') 
    parser.add_argument('--scores_dir', default='scores')
    parser.add_argument('--get_confidence', action='store_true',  help='get_confidence') 
    parser.add_argument('--fit_pwlf', action='store_true',  help='fit piecewise linear function') 
    parser.add_argument('--pwlf_bins', default=0,  help="use binned WCR as targets of pwlf", type=int)
    parser.add_argument('--pwlf_segments', default=5,  help="use binned WCR as targets of pwlf", type=int)
    parser.add_argument('--skip_cem', action='store_true',  help='if true, bare softmax will be used as confidence scores') 
    parser.add_argument('--temp', default=1.0,  help="temperature scaling for softmax outputs", type=float)

    args = parser.parse_args()
    return args

def test_loop(asr_model, cem, dataloader, save_dir=None, limit_batch=-1, skip_cem=False, dataset=""):
    # num_batches = len(dataloader)
    # nce = NCE()
    y_score, y_true = [], []    
    groundtruths = []
    text_predictions = []
    full_groundtruths = []
    with torch.no_grad():
        asr_model.eval()
        cem.eval()
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # (B, T, C) , (B, T, C), (B, )
            signal, signal_len, transcript, transcript_len = batch
            for i in range(len(transcript)):
                seq_len = transcript_len[i].cpu().detach().numpy()
                seq_ids = transcript[i].cpu().detach().numpy()
                reference = ''.join(asr_model._wer.decoding.decode_tokens_to_str(seq_ids[0:seq_len]))
                full_groundtruths.append(reference)                
            conf, conf_gt, b_text_predictions, b_text_references = cem.inference(asr_model, batch, skip_cem)
            #get confidence
            y_score += conf
            y_true += conf_gt
        
            # get text groundtruths
            groundtruths += b_text_references
            # get text prediction
            text_predictions += b_text_predictions

            if limit_batch > 0 and batch_idx >= limit_batch-1:
                break
        asr_model.train()
        asr_model.encoder.freeze()
        asr_model.decoder.freeze() 
        cem.train()
    
    ckpt_name = save_dir
    save_dir += f"/{dataset}"
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    # y_true and y_score are list of lists
    y_t_flatten, y_p_flatten = np.array([y for yin in y_true for y in yin]), np.array([y for yin in y_score for y in yin])

    fig, ax1 = plt.subplots(figsize=plt_figsize)
    precision, recall, thresholds = metrics.precision_recall_curve(y_t_flatten, y_p_flatten)
    avg_precision_pos1 = metrics.average_precision_score(y_t_flatten, y_p_flatten)
    display = metrics.PrecisionRecallDisplay(precision, recall, average_precision=avg_precision_pos1)
    display.plot(ax1)
    plt.savefig(f'{save_dir}/precision_recall.pdf', bbox_inches='tight')

    fig, ax1 = plt.subplots(figsize=plt_figsize)
    precision, recall, thresholds = metrics.precision_recall_curve(y_t_flatten, 1-y_p_flatten, pos_label=0)
    avg_precision_pos0 = metrics.average_precision_score(y_t_flatten, 1-y_p_flatten, pos_label=0)
    display = metrics.PrecisionRecallDisplay(precision, recall, average_precision=avg_precision_pos0, pos_label=0)
    display.plot(ax1)
    plt.savefig(f'{save_dir}/precision_recall_pos0.pdf', bbox_inches='tight')

    fig, ax1 = plt.subplots(figsize=plt_figsize)
    fpr, tpr, thresholds = metrics.roc_curve(y_t_flatten, y_p_flatten)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                      estimator_name=os.path.basename(save_dir))
    display.plot(ax1)
    plt.savefig(f"{save_dir}/roc.pdf", bbox_inches='tight')

    
    # prob_true, prob_pred = calibration_curve(y_t_flatten, y_p_flatten, n_bins=20)
    bin_ids, prob_true, prob_pred, bin_total = calibrate_binning( y_t_flatten, y_p_flatten, nbins=10)
    display = CalibrationDisplay(prob_true, prob_pred, y_p_flatten, )
    ece = np.sum(np.abs(prob_true - prob_pred) * bin_total) / np.sum(bin_total)
    # ece2 = np.mean(np.abs(prob_true - prob_pred))

    fig, ax1 = plt.subplots(figsize=plt_figsize)

    ax2 = ax1.twinx()
    ax2.bar(prob_pred, bin_total/np.sum(bin_total) , width=0.05, color='C2', alpha=0.5)
    ax2.set_ylabel('Probability mass function', color='C2')

    display.plot(ax=ax1)
    ax1.set_ylabel('Fraction of positives', color='C0')
    
    ax1.tick_params(axis='y', labelcolor='C0')
    ax2.tick_params(axis='y', labelcolor='C2')
    plt.savefig(f"{save_dir}/calibration.pdf", bbox_inches='tight')
    
    rmse_wrc = rmse_wcr_score(y_true, y_score)
    rmse_wer = rmse_wer_score(y_score, text_predictions, groundtruths)
    nce = norm_bicross_entropy(y_t_flatten, y_p_flatten)

    if not os.path.exists(f"{save_dir}/metrics.tsv"):
        with open(f"{save_dir}/metrics.tsv", "w") as f:
            f.write(f"ckpt_name\tdataset\troc_auc\tnce\tavg_precision_pos1\tavg_precision_pos0\trmse_wrc\trmse_wer\tece\n")
    with open(f"{save_dir}/metrics.tsv", "a") as f:
        f.write(f"{ckpt_name}\t{dataset}\t{roc_auc:.4f}\t{nce:.4f}\t{avg_precision_pos1:.4f}\t{avg_precision_pos0:.4f}\t{rmse_wrc:.4f}\t{rmse_wer:.4f}\t{ece:.4f}\n")

    sep = ','
    with open(f"{save_dir}/pred_conf.csv", 'w') as f:
        y_score = [ [f"{score:.3f}" for score in yi] for yi in y_score]
        y_true = [ [f"{score:.3f}" for score in yi] for yi in y_true]
        f.write(f"conf{sep}conf_gt{sep}text{sep}text_gt{sep}text_gt_full\n")
        for i in range(len(y_true)):
            f.write(f"{' '.join(y_score[i])}{sep}{' '.join(y_true[i])}{sep}{text_predictions[i]}{sep}{groundtruths[i]}{sep}{full_groundtruths[i]}\n")
       

if __name__ == '__main__':
    args = get_parser()
    
    if args.lang == 'en':
        args.asr_model = "/workspaces/NeMo-outputs/nemo_exp_oct22/Conformer-CTC-Char-small-libri-1khr/checkpoints/Conformer-CTC-Char-small-libri-1khr--val_wer=0.0514-epoch=200-last.ckpt"
    elif args.lang == 'th':
        args.asr_model = "/workspaces/NeMo-outputs/nemo_exp_oct22/Conformer-CTC-Char-th-small-16k/checkpoints/Conformer-CTC-Char-th-small-16k--val_wer=0.1335-epoch=300-last.ckpt"
    else:
        raise NotImplementedError(f"args.lang: {args.lang}")

    if args.fit_pwlf:
        CEM = CEM_PWLF

    if args.cctc:
        
        if args.rnnt:            
            raise ValueError("CCTC in not compatible with RNNT")            
        else:    
            if args.bpe:
                MODEL_ABS = EncDecCCTCModelBPE
            else:
                MODEL_ABS = EncDecCCTCModel

    else:
        if args.rnnt:            
            if args.bpe:        
                MODEL_ABS = EncDecRNNTBPEModel
            else:
                MODEL_ABS = EncDecRNNTModel
        else:
            if args.bpe:        
                MODEL_ABS = EncDecCTCModelBPE
            else:
                MODEL_ABS = EncDecCTCModel

    
    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = MODEL_ABS.restore_from(restore_path=args.asr_model)
    elif args.asr_model.endswith('.ckpt'):
        asr_model = MODEL_ABS.load_from_checkpoint(args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = MODEL_ABS.from_pretrained(model_name=args.asr_model)


    if args.lang == 'en':
        asr_model.setup_training_data(
            train_data_config={
                'sample_rate': 16000,
                'manifest_filepath': "/datasets/libri_new/dev_clean.json",
                # 'manifest_filepath': "/datasets/libri_new/train_clean_100.json", 
                # 'manifest_filepath': "/datasets/libri_new/train_clean_100.json,/datasets/libri_new/train_clean_360.json,/datasets/libri_new/train_other_500.json",
                # 'manifest_filepath': "/datasets/libri_new/dev_clean.json,/datasets/libri_new/dev_other.json",
                'labels': asr_model.decoder.vocabulary,
                'batch_size': 32,
                'normalize_transcripts': True,
                'pin_memory' : True,
                'shuffle': True,
            }
        )
        asr_model._train_dl = asr_model._setup_dataloader_from_config(asr_model._hparams.cfg.train_ds)


    if args.lang == 'th':
        with open_dict(asr_model._hparams.cfg.train_ds):
            asr_model._hparams.cfg.train_ds.bucketing_batch_size = 32
            asr_model._hparams.cfg.train_ds.num_workers =  8

        with open_dict(asr_model._hparams.cfg.validation_ds):
            asr_model._hparams.cfg.validation_ds.batch_size = 8
            asr_model._hparams.cfg.validation_ds.num_workers =  8

        asr_model._train_dl = asr_model._setup_dataloader_from_config(asr_model._hparams.cfg.validation_ds)

    asr_model.preprocessor.featurizer.dither = 0.0
    asr_model.preprocessor.featurizer.pad_to = 0    
    asr_model.eval()
    asr_model = asr_model.cuda()


    blank_index = len(asr_model.decoder.vocabulary)
    space_index = asr_model.decoder.vocabulary.index(' ')

    if args.lang == 'en':
        ckpt = "/workspaces/NeMo-outputs/CEM_libri/ctc_ep200_libri100_fixedagg/b32lr1e-3_logits-onehot-softmax-counter_f2-t10_bce1.0_fixedprob-aggmean_wordaggmean_fc3swish_fixedcounter_incblank/ckpt/model_weights_ep34_nce0.409.pth"

    if args.lang == 'th':
        # ckpt = "/workspaces/NeMo-outputs/CEM_th/ctc_ep300/b32lr1e-3_logits-onehot-softmax_f2-t10_bce1.0_fixedprob-aggmin_wordaggmean_transformer1head/ckpt/model_weights_ep99_nce0.306.pth"
        ckpt = "/workspaces/NeMo-outputs/CEM_th/ctc_ep300_fixedctcagg/b32lr1e-3_logits-onehot-softmax-counter_f2-t10_bce1.0_fixedprob-aggmean_wordaggmean_fc3swish_fixedcounter_incblank/ckpt/model_weights_ep67_nce0.316.pth"

    is_add_transformer = True if 'transformer' in ckpt else False
    ctc_agg, word_agg = re.findall("agg(.{3}|.{4})_", ckpt)
    cnn_ksize = re.findall("cnn(\d)+k", ckpt)
    cnn_ksize = int(cnn_ksize[0]) if len(cnn_ksize) > 0 else 0

    cem = CEM(len(asr_model._hparams.cfg.train_ds.labels) + 1, 1, blank_index, space_index, 
        is_add_onehot=True, is_add_counter=True, is_add_logsoftmax=False, 
        is_add_selfattn=False, inplace_selfattn=False, n_fc_layer=3, is_add_transformer=is_add_transformer, linear_activation='swish',
        is_add_full_selfattn=False, ctc_agg=ctc_agg, word_agg=word_agg, pos_weight=1.0, use_logits=True, use_context_heads=0,
        cnn_kernel_size=cnn_ksize, is_add_softmax=True, temp=args.temp, is_boundary_inclusive=False, skip_blank=False)
    
    state_dict = torch.load(ckpt)
    for key in ["pre_emb.weight", "pre_emb.bias", "emb.in_proj_weight", "emb.in_proj_bias", "emb.out_proj.weight", "emb.out_proj.bias"]:
        if key not in state_dict: continue
        if 'pre_emb' in key:
            state_dict[key.replace("pre_emb", "projection")] = state_dict.pop(key)
        elif 'emb' in key:
            state_dict[key.replace("emb", "self_attn")] = state_dict.pop(key)

    cem.load_state_dict(state_dict, strict=False)
    
    ckpt_name = '.'.join(os.path.basename(ckpt).split(".")[:-1])
    ckpt_folder = os.path.dirname(ckpt)
    base_folder, _ = os.path.split(ckpt_folder)
    save_dir = f'{base_folder}/{ckpt_name}'

    if args.skip_cem:
        
        save_dir = f"{save_dir}_skip-cem-temp{args.temp}_new"
    
    cem = cem.cuda()
    cem.eval()

    if args.fit_pwlf:
        save_dir = f"{save_dir}_pwlf-dev-clean_bins{args.pwlf_bins}"
        pwlf_weight =  f"{save_dir}/pwlf_{args.pwlf_segments}.pk"
        if os.path.exists(pwlf_weight):
            print("###### loading pwlf....")
            with open(pwlf_weight, 'rb') as fb:
                cem.pwlf = pickle.load(fb)
        else:
            print("###### fitting pwlf....")
            cem.fit_pwlf(asr_model, asr_model._train_dl, n_segments=args.pwlf_segments, nbins=args.pwlf_bins)
            os.makedirs(save_dir, exist_ok=True)
            with open(pwlf_weight, 'wb') as fb:
                pickle.dump(cem.pwlf, fb)

    if args.lang == 'en':
        datasets = [
            "/datasets/libri_new/test_clean.json",
            "/datasets/libri_new/test_other.json",
            "/datasets/CommonVoice_dataset/manifest/commonvoice_dev_manifest.json",
            "/datasets/CommonVoice_dataset/manifest/commonvoice_test_manifest.json",
        ]

    if args.lang == 'th':
        datasets = [
        ]

    for dataset in datasets:

        asr_model.setup_test_data(
            test_data_config={
                'sample_rate': 16000,
                'manifest_filepath': dataset,
                'labels': asr_model.decoder.vocabulary,
                'batch_size': 16,
                'normalize_transcripts': True if args.lang == 'en' else False,
                'pin_memory' : True,
                'shuffle': False,
            }
        )
        # asr_model._test_dl = asr_model._setup_dataloader_from_config(asr_model._hparams.cfg.test_ds)

        # dataset = dataset.replace("data-nemo", 'datasets')
        dataset = re.sub(".*data-nemo", "datasets", dataset)
        
        
        test_loop(asr_model, cem, dataloader=asr_model.test_dataloader(), save_dir=save_dir, limit_batch=-1, dataset=dataset, skip_cem=args.skip_cem)
        # test_loop(asr_model, cem, dataloader=asr_model._validation_dl, save_dir=f'{save_dir}/softmax', limit_batch=-1, dataset=dataset, skip_cem=False)

        