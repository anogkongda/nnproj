#!/usr/bin/python
#encoding=utf-8

import os
import sys
import copy
import time 
import yaml
import argparse
import numpy as np
import pdb
import torch
import torch.nn as nn

sys.path.append('./')
from models.model_ctc import *
#from warpctc_pytorch import CTCLoss # use built-in nn.CTCLoss
from utils.data_loader import Vocab, SpeechDataset, SpeechDataLoader

supported_rnn = {'nn.LSTM':nn.LSTM, 'nn.GRU': nn.GRU, 'nn.RNN':nn.RNN}
supported_activate = {'relu':nn.ReLU, 'tanh':nn.Tanh, 'sigmoid':nn.Sigmoid}

parser = argparse.ArgumentParser(description='cnn_lstm_ctc')
parser.add_argument('--conf', default='conf/ctc_config.yaml' , help='conf file with argument of LSTM and training')

def run_epoch(epoch_id, model, data_iter, loss_fn, device, optimizer=None, print_every=20, is_training=True, lids=None):
    if is_training:
        model.train()
    else:
        model.eval()
    
    total_loss = 0
    total_tokens = 0
    total_errs = 0
    cur_loss = 0

    for i, data in enumerate(data_iter):
        inputs, input_sizes, targets, target_sizes, utt_list = data
        B,T,_ = inputs.shape
        if lids is not None:
            xx = lids.repeat(B,T,1)
            inputs = torch.cat((inputs,xx), dim=-1)
        inputs = inputs.to(device)
        input_sizes = input_sizes.to(device)
        targets = targets.to(device)
        target_sizes = target_sizes.to(device)
       
        out = model(inputs)
        out_len, batch_size, _ = out.size()
        input_sizes = (input_sizes * out_len).long()
        loss = loss_fn(out, targets, input_sizes, target_sizes)
        loss /= batch_size
        cur_loss += loss.item()
        total_loss += loss.item()
        prob, index = torch.max(out, dim=-1)
        batch_errs, batch_tokens = model.compute_wer(index.transpose(0,1).cpu().numpy(), input_sizes.cpu().numpy(), targets.cpu().numpy(), target_sizes.cpu().numpy())
        total_errs += batch_errs
        total_tokens += batch_tokens

        if (i + 1) % print_every == 0 and is_training:
            print('Epoch = %d, step = %d, cur_loss = %.4f, total_loss = %.4f, total_wer = %.4f' % (epoch_id, 
                                    i+1, cur_loss / print_every, total_loss / (i+1), total_errs / total_tokens ))
            cur_loss = 0
        
        if is_training:    
            optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 400)
            optimizer.step()
    average_loss = total_loss / (i+1)
    training = "Train" if is_training else "Valid"
    print("Epoch %d %s done, total_loss: %.4f, total_wer: %.4f" % (epoch_id, training, average_loss, total_errs / total_tokens))
    return 1-total_errs / total_tokens, average_loss

class Config(object):
    batch_size = 4
    dropout = 0.1

def main(conf):
    opts = Config()
    for k, v in conf.items():
        setattr(opts, k, v)
        print('{:50}:{}'.format(k, v))

    device = torch.device('cuda') if opts.use_gpu else torch.device('cpu')
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    if opts.use_gpu:
        torch.cuda.manual_seed(opts.seed)
    
    data_file = opts.data_file
    language = data_file.split("/")[1]
    language_dict = {}
    with open(opts.language_order) as f:
        for idx,line in enumerate(f.readlines()):
            line = line.strip()
            language_dict[line] = idx
    language_id = language_dict[language]


    train_scp = "/train/feats.scp"
    train_lab = "/train/lab.txt"
    valid_scp = "/dev/feats.scp"
    valid_lab = "/dev/lab.txt"
    vocab_f = "/units"
    if opts.universal:
        vocab_f = "/all_units"
    #Data Loader
    vocab = Vocab(data_file + vocab_f)
    train_dataset = SpeechDataset(vocab, data_file + train_scp, data_file + train_lab, opts)
    dev_dataset = SpeechDataset(vocab, data_file + valid_scp, data_file + valid_lab, opts)
    train_loader = SpeechDataLoader(train_dataset, batch_size=opts.batch_size, shuffle=opts.shuffle_train, num_workers=opts.num_workers)
    dev_loader = SpeechDataLoader(dev_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    if opts.language_one_hot:
        # add size of one-hot label
        opts.rnn_input_size = opts.rnn_input_size + len(language_dict.items())
        lid = torch.zeros(len(language_dict.items()))
        lid[language_id] = 1

    #Define Model
    rnn_type = supported_rnn[opts.rnn_type]
    rnn_param = {"rnn_input_size":opts.rnn_input_size, "rnn_hidden_size":opts.rnn_hidden_size, "rnn_layers":opts.rnn_layers, 
                    "rnn_type":rnn_type, "bidirectional":opts.bidirectional, "batch_norm":opts.batch_norm}
    
    num_class = vocab.n_words
    opts.output_class_dim = vocab.n_words
    drop_out = opts.drop_out
    add_cnn = opts.add_cnn
    
    cnn_param = {}
    channel = eval(opts.channel)
    kernel_size = eval(opts.kernel_size)
    stride = eval(opts.stride)
    padding = eval(opts.padding)
    pooling = eval(opts.pooling)
    activation_function = supported_activate[opts.activation_function]
    cnn_param['batch_norm'] = opts.batch_norm
    cnn_param['activate_function'] = activation_function
    cnn_param["layer"] = []
    for layer in range(opts.layers):
        layer_param = [channel[layer], kernel_size[layer], stride[layer], padding[layer]]
        if pooling is not None:
            layer_param.append(pooling[layer])
        else:
            layer_param.append(None)
        cnn_param["layer"].append(layer_param)

    model = CTC_Model(add_cnn=add_cnn, cnn_param=cnn_param, rnn_param=rnn_param, num_class=num_class, drop_out=drop_out)
    
    # Load language specific model
    if opts.resume != '' :
        print("Load ckp from {}".format(opts.resume))
        package = torch.load(opts.resume)
        state_dict = package['state_dict']
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model.state_dict().keys()}
        prefix = "fc_list." + str(language_id)
        if opts.universal:
            prefix = "fc_list.0"
        language_softmax_dict = {k:v for k,v in state_dict.items() if k.startswith(prefix)}
        for k,v in language_softmax_dict.items():
            new_key = k.replace(prefix,"fc")
            pretrained_dict[new_key] = v
        model.load_state_dict(pretrained_dict)
        
    
    model = model.to(device)
    num_params = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
    print("Number of parameters %d" % num_params)
    for idx, m in enumerate(model.children()):
        print(idx, m)
    
    #Training
    init_lr = opts.init_lr
    num_epoches = opts.num_epoches
    end_adjust_acc = opts.end_adjust_acc
    decay = opts.lr_decay
    weight_decay = opts.weight_decay
    batch_size = opts.batch_size
    
    params = { 'num_epoches':num_epoches, 'end_adjust_acc':end_adjust_acc, 'mel': opts.mel, 'seed':opts.seed,
                'decay':decay, 'learning_rate':init_lr, 'weight_decay':weight_decay, 'batch_size':batch_size,
                'feature_type':opts.feature_type, 'n_feats': opts.feature_dim }
    print(params)
    
    loss_fn = nn.CTCLoss(reduction='sum',zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    
    #visualization for training
    
    count = 0
    learning_rate = init_lr
    loss_best = 1000
    loss_best_true = 1000
    adjust_rate_flag = False
    stop_train = False
    adjust_time = 0
    acc_best = 0
    start_time = time.time()
    loss_results = []
    dev_loss_results = []
    dev_cer_results = []
    
    while not stop_train:
        if count >= num_epoches:
            break
        count += 1
        
        if adjust_rate_flag:
            learning_rate *= decay
            adjust_rate_flag = False
            for param in optimizer.param_groups:
                param['lr'] *= decay
        
        print("Start training epoch: %d, learning_rate: %.5f" % (count, learning_rate))
        
        train_acc, loss = run_epoch(count, model, train_loader, loss_fn, device, optimizer=optimizer, print_every=opts.verbose_step, is_training=True, lids=lid if opts.language_one_hot else None)
        loss_results.append(loss)
        acc, dev_loss = run_epoch(count, model, dev_loader, loss_fn, device, optimizer=None, print_every=opts.verbose_step, is_training=False, lids=lid if opts.language_one_hot else None)
        print("loss on dev set is %.4f" % dev_loss)
        dev_loss_results.append(dev_loss)
        dev_cer_results.append(acc)
        
        #adjust learning rate by dev_loss
        if dev_loss < (loss_best - end_adjust_acc):
            loss_best = dev_loss
            loss_best_true = dev_loss
            adjust_rate_count = 0
            model_state = copy.deepcopy(model.state_dict())
            op_state = copy.deepcopy(optimizer.state_dict())
        elif (dev_loss < loss_best + end_adjust_acc):
            adjust_rate_count += 1
            if dev_loss < loss_best and dev_loss < loss_best_true:
                loss_best_true = dev_loss
                model_state = copy.deepcopy(model.state_dict())
                op_state = copy.deepcopy(optimizer.state_dict())
        else:
            adjust_rate_count = 10
        
        if acc > acc_best:
            acc_best = acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_op_state = copy.deepcopy(optimizer.state_dict())

        print("adjust_rate_count:"+str(adjust_rate_count))
        print('adjust_time:'+str(adjust_time))

        if adjust_rate_count == 10:
            adjust_rate_flag = True
            adjust_time += 1
            adjust_rate_count = 0
            if loss_best > loss_best_true:
                loss_best = loss_best_true
            model.load_state_dict(model_state)
            optimizer.load_state_dict(op_state)

        if adjust_time == 8:
            stop_train = True
        
        time_used = (time.time() - start_time) / 60
        print("epoch %d done, cv acc is: %.4f, time_used: %.4f minutes" % (count, acc, time_used))
        
        x_axis = range(count)
        y_axis = [loss_results[0:count], dev_loss_results[0:count], dev_cer_results[0:count]]
        
    print("End training, best dev loss is: %.4f, acc is: %.4f" % (loss_best, acc_best))
    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_op_state)
    save_dir = os.path.join(opts.checkpoint_dir, opts.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best_path = os.path.join(save_dir, 'ctc_best_model.pkl')
    params['epoch']=count

    torch.save(CTC_Model.save_package(model, optimizer=optimizer, epoch=params, loss_results=loss_results, dev_loss_results=dev_loss_results, dev_cer_results=dev_cer_results), best_path)

if __name__ == '__main__':
    args = parser.parse_args()
    try:
        config_path = args.conf
        conf = yaml.safe_load(open(config_path, 'r'))
    except:
        print("No input config or config file missing, please check.")
        sys.exit(1)
    main(conf)
