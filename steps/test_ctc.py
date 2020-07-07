#!/usr/bin/python
#encoding=utf-8

import os
import time
import sys
import torch
import yaml
import argparse
import torch.nn as nn
import pdb
sys.path.append('./')
from models.model_ctc import *
from utils.ctcDecoder import GreedyDecoder, BeamDecoder
from utils.data_loader import Vocab, SpeechDataset, SpeechDataLoader
from steps.train_ctc import Config

parser = argparse.ArgumentParser()
parser.add_argument('--conf', default='/mnt/lustre/sjtu/home/zkz01/tools/CTC_pytorch/pipeline/conf/ctc_config.yaml',help='conf file for training')

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
    separator = opts.separator if opts.separator else " "
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    
    model_path = os.path.join(opts.checkpoint_dir, opts.exp_name, 'ctc_best_model.pkl')
    package = torch.load(model_path)
    
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
        vocab_file = opts.data_file +"/all_units"
    vocab = Vocab(vocab_file)
    num_class = vocab.n_words
    test_dataset = SpeechDataset(vocab, opts.test_scp_path, opts.test_lab_path, opts)
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
    
    if decoder_type == 'Greedy':
        decoder  = GreedyDecoder(vocab.index2word, space_idx=-1, blank_index=0)
    else:
        decoder = BeamDecoder(vocab.index2word, beam_width=beam_width, blank_index=0, space_idx=-1, lm_path=opts.lm_path, lm_alpha=opts.lm_alpha)    
   
    total_wer = 0
    total_cer = 0
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
            decoded = decoder.decode(probs, input_sizes.numpy().tolist())
            
            targets, target_sizes = targets.numpy(), target_sizes.numpy()
            labels = []
            for i in range(len(targets)):
                label = [ vocab.index2word[num] for num in targets[i][:target_sizes[i]]]
                labels.append(' '.join(label))

            for x in range(len(targets)):
                print("origin : " + labels[x])
                print("decoded: " + decoded[x])
            cer = 0
            wer = 0
            for x in range(len(labels)):
                cer += decoder.cer(decoded[x], labels[x])
                wer += decoder.wer(decoded[x], labels[x], separator)
                decoder.num_word += len(labels[x].split(separator))
                decoder.num_char += len(labels[x])
            total_cer += cer
            total_wer += wer
    CER = (float(total_cer) / decoder.num_char)*100
    WER = (float(total_wer) / decoder.num_word)*100
    print("Character error rate on test set: %.4f" % CER)
    print("Word error rate on test set: %.4f" % WER)
    end = time.time()
    time_used = (end - start) / 60.0
    print("time used for decode %d sentences: %.4f minutes." % (len(test_dataset), time_used))

if __name__ == "__main__":
    test()
