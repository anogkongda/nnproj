#!/usr/bin/python
#encoding=utf-8

import os
import time
import sys
import torch
import yaml
import argparse
import torch.nn as nn
import numpy as np
sys.path.append('./')
from models.model_ctc import *
from utils.ctcDecoder import GreedyDecoder, BeamDecoder
from utils.data_loader import Vocab, SpeechDataset, SpeechDataLoader
from steps.train_ctc import Config
import pdb
parser = argparse.ArgumentParser()
parser.add_argument('--conf', default='/mnt/lustre/sjtu/home/zkz01/tools/CTC_pytorch/pipeline/conf/ctc_config.yaml',help='conf file for training')


def exist_kwd(decoded, keywords):
    prob_mat = np.zeros((len(decoded), len(keywords)))
    for i,utt in enumerate(decoded):
        words = ''.join(utt.split()).split("<eow>")
        for j,kw in enumerate(keywords):
            if kw in words:
                prob_mat[i,j] = 1
    return prob_mat

def soft_kwd(decoded, keywords):
    prob_mat = np.zeros((len(decoded), len(keywords)))
    for i,utts_with_prob in enumerate(decoded):
        utts,prob = utts_with_prob
        utts = [''.join(utt.split()).split("<eow>") for utt in utts]
        for j,kw in enumerate(keywords):
            max_pos_prob = 0.0
            max_neg_prob = 0.0
            for idx,utt in enumerate(utts):
                if kw in utt and prob[idx] >max_pos_prob:
                    max_pos_prob = prob[idx]
                elif kw not in utt and prob[idx] > max_neg_prob:
                    max_neg_prob = prob[idx]
            prob_mat[i,j] = max_pos_prob / (max_pos_prob + max_neg_prob + 1e-8)
                    
    return prob_mat



def test():
    args = parser.parse_args()
    try:
        conf = yaml.safe_load(open(args.conf,'r'))
    except:
        print("Config file not exist!")
        sys.exit(1)    
    
    opts = Config()
    for k,v in conf.items():
        setattr(opts, k, v)
        print('{:50}:{}'.format(k, v))

    use_cuda = opts.use_gpu
    # use_cuda = False
    separator = opts.separator if opts.separator else " "
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    
    model_path = os.path.join(opts.checkpoint_dir, opts.exp_name, 'ctc_best_model.pkl')
    package = torch.load(model_path, map_location=device)
    
    rnn_param = package["rnn_param"]
    add_cnn = package["add_cnn"]
    cnn_param = package["cnn_param"]
    feature_type = package['epoch']['feature_type']
    n_feats = package['epoch']['n_feats']
    drop_out = package['_drop_out']
    mel = opts.mel

    beam_width = opts.beam_width
    lm_alpha = opts.lm_alpha
    decoder_type =  opts.decode_type
    vocab_file = opts.data_file + "/units"
    if opts.universal:
        vocab_file = opts.data_file + "/all_units"
    keywords = []
    with open(opts.keyword_path, 'r') as f:
        for kw in f.readlines():
            kw = kw.rstrip("\n")
            keywords.append(kw)
    
    pos_probs = {}
    neg_probs = {}
    for kw in keywords:
        pos_probs[kw] = []
        neg_probs[kw] = []

    
    vocab = Vocab(vocab_file)
    num_class = vocab.n_words
    test_dataset = SpeechDataset(None, opts.test_scp_path, opts.test_kws_lab_path, opts)
    test_loader = SpeechDataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers, pin_memory=False)
    
    model = CTC_Model(rnn_param=rnn_param, add_cnn=add_cnn, cnn_param=cnn_param, num_class=num_class, drop_out=drop_out)
    model.to(device)


    language = opts.data_file.split("/")[1]
    language_dict = {}
    with open(opts.language_order) as f:
        for idx,line in enumerate(f.readlines()):
            line = line.strip()
            language_dict[line] = idx
    language_id = language_dict[language]


    if opts.from_multi:
        print("Load from multi")
        state_dict = package['state_dict']
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model.state_dict().keys()}
        prefix = "fc_list." + str(language_id)
        language_softmax_dict = {k:v for k,v in state_dict.items() if k.startswith(prefix)}
        for k,v in language_softmax_dict.items():
            new_key = k.replace(prefix,"fc")
            pretrained_dict[new_key] = v
        
        model.load_state_dict(pretrained_dict)
    else:
        model.load_state_dict(package['state_dict'])
    
    model.eval()
    if opts.language_one_hot:
        # add size of one-hot label
        lid = torch.zeros(len(language_dict.items()))
        lid[language_id] = 1
    
    '''
    Decode   Initialize: keywords, blank_index, beam_width

    inputs: probs, length

    outputs: probs for each keyword

    '''
    
    decoder = BeamDecoder(vocab.index2word, beam_width=beam_width, blank_index=0, space_idx=-1, lm_path=opts.lm_path, lm_alpha=opts.lm_alpha)    
   
    utt_idx = 0
    start = time.time()
    with torch.no_grad():
        for data in test_loader:
            inputs, input_sizes, targets, target_sizes, utt_list = data 
            if opts.language_one_hot:
                B,T,_ = inputs.shape
                xx = lid.repeat(B,T,1)
                inputs = torch.cat((inputs,xx), dim=-1)
            
            inputs = inputs.to(device)
            #rnput_sizes = input_sizes.to(device) 
            #target = target.to(device)
            #target_sizes = target_sizes.to(device)
            
            probs = model(inputs)

            max_length = probs.size(0)
            input_sizes = (input_sizes * max_length).long()

            probs = probs.cpu()

            if decoder_type == "soft":
                decoded = decoder.decode(probs, input_sizes.numpy().tolist(),n_best=True)
                prob_mat = soft_kwd(decoded,keywords)
            else:

                decoded = decoder.decode(probs, input_sizes.numpy().tolist())
                # output existence for each keyword
                prob_mat = exist_kwd(decoded, keywords)
            # target is a 0-1 matrix
            targets, target_sizes = targets.numpy(), target_sizes.numpy()

            for i in range(len(decoded)):
                for j,kw in enumerate(keywords):
                    if targets[i,j] == 1:
                        pos_probs[kw].append(prob_mat[i,j])
                    else:
                        neg_probs[kw].append(prob_mat[i,j])
            utt_idx += len(decoded)
            print("Processed {}/{} utterances.".format(utt_idx, len(test_dataset)))

    expdir = opts.checkpoint_dir + opts.exp_name
    print("Output to {}".format(expdir))
    threshold = 0.5
    FPs = {}
    TPs = {}
    for item in pos_probs.items():
        kw = item[0]
        probs = item[1]
        probs = np.array(probs)
        TPs[kw] = len(probs[probs>=threshold])
        with open(expdir+"/" + kw +".pos", 'w') as f:
            for prob in probs:
                f.write(str(prob) + "\n")
        
    for item in neg_probs.items():
        kw = item[0]
        probs = item[1]
        probs = np.array(probs)
        FPs[kw] = len(probs[probs>=threshold])
        with open(expdir+"/" + kw +".neg", 'w') as f:
            for prob in probs:
                f.write(str(prob) + "\n")
    
    for kw in keywords:
        recall = TPs[kw] / (len(pos_probs[kw])+1e-8)
        precision = TPs[kw] / (TPs[kw] + FPs[kw] + 1e-8)
        print("For keyword {} of threshold {}: Precision {}, Recall {}, F1 {}".format(kw, str(threshold), recall,precision, 2 * (precision * recall) / (precision + recall + 1e-8)))

    print("kws decode method: {}".format(decoder_type))
    end = time.time()
    time_used = (end - start) / 60.0
    print("time used for decode %d sentences: %.4f minutes." % (len(test_dataset), time_used))

if __name__ == "__main__":
    test()



