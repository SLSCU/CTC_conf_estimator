import pickle
from re import L
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from argparse import ArgumentParser
from CEM_module import CEM
import logging
from nemo.collections.asr.models import EncDecCTCModel, EncDecCCTCModel, EncDecCTCModelBPE, EncDecRNNTBPEModel, EncDecRNNTModel
import torch
from tqdm import tqdm
from omegaconf import open_dict
import numpy as np
import torch.nn.functional as F
import datetime
from cem_utils import NCE, norm_bicross_entropy
from sklearn.metrics import roc_auc_score

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
    parser.add_argument('--sr', default=8000,  help='Sampling rate.', type=int) 
    parser.add_argument('--retokenize', action='store_true',  help='retokenize') 
    parser.add_argument('--scores_dir', default='scores')
    parser.add_argument('--get_confidence', action='store_true',  help='get_confidence') 
    args = parser.parse_args()
    return args


def train_loop(asr_model, cem, dataloader, max_step=-1, save_dir=None, save_step=-1):
    optimizer = cem.configure_optimizers()
    size = len(dataloader)
    loss_log = f'{save_dir}/loss.log'
    cum_loss = 0.

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        loss = cem.training_step(asr_model, batch, batch_idx)
        loss.backward()
        optimizer.step()
        cum_loss += loss.item()
        if batch_idx % 100 == 0:
            cum_loss, current = cum_loss/100, batch_idx
            with open(loss_log, 'a') as loss_log_file:
                print_str = f"{datetime.datetime.now()} {batch_idx} loss: {cum_loss:>7f} [{current:>5d}/{size:>5d}]"
                print(print_str, file=loss_log_file)
                print(print_str)
            cum_loss = 0.
            
        if max_step > 0 and batch_idx > max_step:
            break
    
        if save_dir != None and save_step != -1 and batch_idx % save_step == 0:
            torch.save(cem.state_dict(), f'{save_dir}/model_weights_ep{batch_idx}.pth')


def test_loop(asr_model, cem, dataloader, save_dir=None, limit_batch=-1, skip_cem=False):
    y_score, y_true = [], []
    with torch.no_grad():
        size = len(dataloader)
        asr_model.eval()
        cem.eval()
        for batch_idx, batch in tqdm(enumerate(dataloader), total=size, disable=True):
            # list of list (B, T)
            conf, conf_gt_flatten, text_predictions, text_references = cem.inference(asr_model, batch)
            for bi in range(len(conf)):
                y_score += conf[bi]
                y_true += conf_gt_flatten[bi]
            # nce.update(w_gt, pred_conf, feats_len) 
            if limit_batch > 0 and batch_idx + 1 >= limit_batch:
                break
        asr_model.train()
        cem.train()
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()
    
    nce = norm_bicross_entropy(y_true, y_score)
    auc = roc_auc_score(y_true, y_score, average='macro')
    return nce, auc
    

def main(asr_model, cem, save_dir, start_ep=0):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/ckpt", exist_ok=True)
    with open(f"{save_dir}/param.txt", 'w') as f:
        print(cem, file=f)
    print('start!')

    for epoch in range(start_ep, 100):
        print("epoch", epoch)
        train_loop(asr_model, cem, save_dir=save_dir, dataloader=asr_model._train_dl, max_step=10_000)
        nce, auc = test_loop(asr_model, cem,  dataloader=asr_model._validation_dl, save_dir=save_dir)
        save_path = f'{save_dir}/ckpt/model_weights_ep{epoch}_nce{nce:>0.3f}.pth'
        torch.save(cem.state_dict(), save_path)
        with open(f'{save_dir}/log.tsv', 'a') as f:
            print(f"{datetime.datetime.now()}\t{auc:>0.3f}\t{nce:0.3f}\t{save_path}", file=f)


if __name__ == '__main__':
    args = get_parser()
    # args.cctc = True
    args.asr_model = "/workspaces/NeMo-outputs/nemo_exp_oct22/Conformer-CTC-Char-small-libri-1khr/checkpoints/Conformer-CTC-Char-small-libri-1khr--val_wer=0.0514-epoch=200-last.ckpt"

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

    # labels = [ asr_model.decoder.vocabulary[i] for i in range(len(asr_model.decoder.vocabulary)) ]
    # print(labels)

    # with open_dict(asr_model._hparams.cfg.train_ds):
    #     asr_model._hparams.cfg.train_ds.labels = labels
    # with open_dict(asr_model._hparams.validation_ds):
    #     asr_model._hparams.validation_ds.labels = labels

    asr_model.setup_training_data(
        train_data_config={
            'sample_rate': 16000,
            'manifest_filepath': "/datasets/libri_new/train_clean_100.json",
            'labels': asr_model.decoder.vocabulary,
            'batch_size': 32,
            'normalize_transcripts': False,
            'pin_memory' : True,
            'shuffle': True,
            'num_workers': 8,
        }
    )

    asr_model.setup_validation_data(
        val_data_config={
            'sample_rate': 16000,
            'manifest_filepath': "/datasets/libri_new/dev_clean.json,/datasets/libri_new/dev_other.json",
            'labels': asr_model.decoder.vocabulary,
            'batch_size': 32,
            'normalize_transcripts': False,
            'pin_memory' : True,
            'shuffle': False,
        }
    )

    asr_model._train_dl = asr_model._setup_dataloader_from_config(asr_model._hparams.cfg.train_ds)
    asr_model._validation_dl = asr_model._setup_dataloader_from_config(asr_model._hparams.cfg.validation_ds)
    # asr_model.preprocessor.featurizer.dither = 0.0
    # asr_model.preprocessor.featurizer.pad_to = 0    
    asr_model.train()
    asr_model.encoder.freeze()
    asr_model.decoder.freeze()
    from nemo.collections.asr.modules import SpectrogramAugmentation
    asr_model.spec_augmentation = SpectrogramAugmentation(
        freq_masks = 2,
        time_masks = 10,
        freq_width = 27,
        time_width = 0.05,
    )

    blank_index = len(asr_model._hparams.cfg.train_ds.labels)
    space_index = asr_model._hparams.cfg.train_ds.labels.index(' ')
    start_ep = 0

    cem = CEM(len(asr_model._hparams.cfg.train_ds.labels) + 1, 1, blank_index, space_index, 
        is_add_onehot=True, is_add_counter=True, is_add_logsoftmax=False, 
        is_add_selfattn=False, inplace_selfattn=False, n_fc_layer=3, is_add_transformer=True, linear_activation='swish',
        is_add_full_selfattn=False, ctc_agg='max', word_agg='mean', pos_weight=1.0, use_logits=True, use_context_heads=0,
        cnn_kernel_size=0, is_add_softmax=True, temp=1, is_boundary_inclusive=False, skip_blank=False)
    
    save_dir = "logdir/"
    load_ckpt = ""

    if load_ckpt != "":
        cem.load_state_dict(torch.load(load_ckpt))
        # save_dir = os.path.dirname(load_ckpt)
        start_ep = int(load_ckpt.split("/")[-1].split("ep")[-1].split("_")[0]) + 1
    
    print(save_dir)


    if torch.cuda.is_available():
        asr_model = asr_model.cuda()
        cem = cem.cuda()

    main(asr_model, cem, save_dir, start_ep=start_ep)

    