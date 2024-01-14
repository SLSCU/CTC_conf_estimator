import pickle
from re import L, S
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from argparse import ArgumentParser
from CEM_module import CEM
import logging
from nemo.collections.asr.models import EncDecCTCModel, EncDecCCTCModel, EncDecCTCModelBPE, EncDecCCTCModelBPE, EncDecRNNTBPEModel, EncDecRNNTModel
import torch
from tqdm import tqdm
from omegaconf import open_dict
import numpy as np
import torch.nn.functional as F
import datetime
from cem_utils import NCE, norm_bicross_entropy
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

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
    parser.add_argument('--lang', default='th',  help='language')  
    parser.add_argument('--retokenize', action='store_true',  help='retokenize') 
    parser.add_argument('--scores_dir', default='scores')
    parser.add_argument('--get_confidence', action='store_true',  help='get_confidence') 
    args = parser.parse_args()
    return args



def tune_loop(asr_model, cem, dataloader, limit_batch=-1, skip_cem=False):
    y_score, y_true = [], []
    log_probs = torch.empty( (0, 94), dtype=torch.float32)
    with torch.no_grad():
        asr_model.eval()
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            # (B, T, C) , (B, T, C), (B, )
            features, w_gt, feats_len = cem.get_features(asr_model, batch)
            B, T, C = features.shape
            
            for bs in range(features.shape[0]):
                pred_conf = features[bs, :feats_len[bs], :].view(-1, C)
                log_probs = torch.cat([log_probs, pred_conf ], dim=0)
                y_true += w_gt[bs, :feats_len[bs]].tolist()
            
            if limit_batch > 0 and batch_idx > limit_batch:
                break

        asr_model.train()
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()   

    y_score, _ = torch.softmax(log_probs, dim=-1).max(dim=-1)
    y_score = y_score.tolist()
    plt.hist(y_score, bins=50)
    plt.title('softmax histrogram')
    plt.savefig('/workspaces/NeMo-outputs/CEM/logprob hist.png')
    nce = norm_bicross_entropy(y_score, y_true)
    auc = roc_auc_score(y_true, y_score, average='macro')
    with open('/workspaces/NeMo-outputs/CEM/temp.tsv', 'a') as f:
        print(f"{datetime.datetime.now()}\t{auc}\t{nce}\t0", file=f)
        
    for temp in np.arange(0.1, 5, 0.1):
        y_score, _ = torch.softmax(log_probs/temp, dim=-1).max(dim=-1)
        y_score = y_score.tolist()    
        nce = norm_bicross_entropy(y_score, y_true)
        auc = roc_auc_score(y_true, y_score, average='macro')
        
        with open('/workspaces/NeMo-outputs/CEM/temp.tsv', 'a') as f:
            print(f"{datetime.datetime.now()}\t{auc}\t{nce}\t{temp}", file=f)


def main(asr_model, cem):
    print('start!')
    tune_loop(asr_model, cem,  dataloader=asr_model._validation_dl,  limit_batch=-1)


if __name__ == '__main__':
    args = get_parser()
    args.cctc = True
    # args.asr_model = "nemo_exp_all/nemo_experiments_ConformerLarge/Conformer-CCTC-BPE-300_ct1.0_ctctc0.0mf_clipgrad1.0_warmup25k_recspecaug/checkpoints/Conformer-CCTC-BPE-300_ct1.0_ctctc0.0mf_clipgrad1.0_warmup25k_recspecaug--val_wer=0.0872-epoch=163.ckpt"
    args.asr_model = "nemo_exp_all/nemo_experiments_Conformer/Conformer-CCTC-Char-th-mediumO0_contcv3-lr0.01_ct0.5-1.0_warm10k/checkpoints/Conformer-CCTC-Char-th-mediumO0_contcv3-lr0.01_ct0.5-1.0_warm10k--val_wer=0.11-epoch=103-v1.ckpt"

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

    labels = [ asr_model.decoder.vocabulary[i] for i in range(len(asr_model.decoder.vocabulary)) ]
    print(labels)
    with open_dict(asr_model._hparams.train_ds):
        asr_model._hparams.train_ds.labels = labels
    with open_dict(asr_model._hparams.validation_ds):
        asr_model._hparams.validation_ds.labels = labels
    # asr_model._train_dl = asr_model._setup_dataloader_from_config(asr_model._hparams.train_ds)
    # asr_model._validation_dl = asr_model._setup_dataloader_from_config(asr_model._hparams.validation_ds)
    asr_model.setup_training_data(
        train_data_config={
            'sample_rate': 8000,
            'manifest_filepath': "data-nemo-dev/dev_sample_s_filter_num_norm_retok-deep.json,data-nemo-dev/cv7.0-dev_uniq_num_norm_retok-deep.json,data-nemo-dev/cv7.0-dev_num_norm_retok-deep.json,data-nemo-dev/cmkl-dev_num_norm_retok-deep.json",
            'labels': asr_model.decoder.vocabulary,
            'batch_size': 16,
            'normalize_transcripts': False,
            'pin_memory' : True,
            'shuffle': True,
        }
    )

    asr_model.setup_validation_data(
        val_data_config={
            'sample_rate': 8000,
            'manifest_filepath': "data-nemo-dev/dev_sample_s_filter_num_norm_retok-deep2.json,data-nemo-dev/cv7.0-dev_uniq_num_norm_retok-deep2.json,data-nemo-dev/cv7.0-dev_num_norm_retok-deep2.json,data-nemo-dev/cmkl-dev_num_norm_retok-deep2.json",
            'labels': asr_model.decoder.vocabulary,
            'batch_size': 16,
            'normalize_transcripts': False,
            'pin_memory' : True,
            'shuffle': False,
        }
    )
    # asr_model.preprocessor.featurizer.dither = 0.0
    # asr_model.preprocessor.featurizer.pad_to = 0    
    asr_model.train()
    asr_model.encoder.freeze()
    asr_model.decoder.freeze()    
    from nemo.collections.asr.modules import SpectrogramAugmentation
    asr_model.spec_augmentation = SpectrogramAugmentation(
        freq_masks = 2,
        time_masks = 5,
        freq_width = 27,
        time_width = 0.05,
    )

    blank_index = len(asr_model._hparams.train_ds.labels)
    space_index = blank_index - 1 
    cem = CEM(94, 1, blank_index, space_index)
    
    # cem.load_state_dict(torch.load( f'{save_dir}/model_weights_2.pth'))    

    if torch.cuda.is_available():
        asr_model = asr_model.cuda()
        cem = cem.cuda()

    main(asr_model, cem)    

    