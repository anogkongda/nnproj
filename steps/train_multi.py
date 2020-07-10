#!/usr/bin/python
#encoding=utf-8

import os
import sys
import copy
import time 
import yaml
import argparse
import numpy as np
import copy
import torch
import torch.nn as nn
from collections.abc import Iterable
sys.path.append('./')
from models.model_ctc import *
#from warpctc_pytorch import CTCLoss # use built-in nn.CTCLoss
from utils.data_loader import Vocab, SpeechDataset, SpeechDataLoader, UnlabelSpeechDataLoader, UnlabelSpeechDataset
import pdb

supported_rnn = {'nn.LSTM':nn.LSTM, 'nn.GRU': nn.GRU, 'nn.RNN':nn.RNN}
supported_activate = {'relu':nn.ReLU, 'tanh':nn.Tanh, 'sigmoid':nn.Sigmoid}

parser = argparse.ArgumentParser(description='cnn_lstm_ctc')
parser.add_argument('--conf', default='conf/multi_config.yaml' , help='conf file with argument of LSTM and training')

def gen_batch(dataloader):
    for i,data in enumerate(dataloader):
        yield data

def language_one_hot_cat(inputs, real_dataset_id, num_sup_dataset):
    B,T,_ = inputs.shape
    device = inputs.device
    lids = torch.zeros((B,T,num_sup_dataset))
    lids[:,:,real_dataset_id] = 1
    inputs = torch.cat((inputs, lids), dim=-1)
    return inputs.to(device)

def sup_get_data(data, device):
    inputs, input_sizes, targets, target_sizes, utt_list = data
    targets = targets.to(device)
    target_sizes = target_sizes.to(device)
    inputs = inputs.to(device)
    input_sizes = input_sizes.to(device)
    return inputs, input_sizes,targets, target_sizes, utt_list



'''
common input: model,loss_fn, data, device, dataset_idxs
common output: loss, out, batch_size, input_sizes, targets, target_sizes
'''
def normal_sup_training(model,loss_fn, data, device, dataset_idx, num_sup_dataset):
    inputs, input_sizes, targets, target_sizes, utt_list = sup_get_data(data, device)
    out = model(inputs, set_id=dataset_idx, mmen_grl=False)
    out_len, batch_size, _ = out.size()
    input_sizes = (input_sizes * out_len).long()
    loss = loss_fn(out, targets, input_sizes, target_sizes)
    return loss, out, batch_size, input_sizes, targets, target_sizes



def dat_sup_training(model,loss_fn, data, device, dataset_idx, num_sup_dataset):
    inputs, input_sizes, targets, target_sizes, utt_list = sup_get_data(data, device)
    out,language_out = model(inputs, set_id = dataset_idx)
    out_len, batch_size, _ = out.size()
    input_sizes = (input_sizes * out_len).long()

    language_target = torch.zeros(batch_size)
    language_target = language_target.fill_(dataset_idx)
    language_target = language_target.view(-1).long().to(device)
    ce = nn.CrossEntropyLoss(ignore_index=-1,reduction='sum')
    language_loss = ce(language_out, language_target)

    asr_loss = loss_fn(out, targets, input_sizes, target_sizes)
    loss = asr_loss + language_loss
    return loss, out, batch_size, input_sizes, targets, target_sizes



def dat_semi_training(model,loss_fn, data, device, dataset_idx, num_sup_dataset):
    inputs, input_sizes, utt_list = data
    real_dataset_id = dataset_idx - num_sup_dataset
    out,language_out = model(inputs, set_id = real_dataset_id)
    out_len, batch_size, _ = out.size()
    input_sizes = (input_sizes * out_len).long()

    language_target = torch.zeros(batch_size)
    language_target = language_target.fill_(real_dataset_id)
    language_target = language_target.view(-1).long().to(device)
    ce = nn.CrossEntropyLoss(ignore_index=-1,reduction='sum')
    language_loss = ce(language_out, language_target)
    return language_loss, out, batch_size, input_sizes, 0,0


def mme_semi_training(model,loss_fn, data, device, dataset_idx, num_sup_dataset):
    inputs, input_sizes, utt_list = data
    real_dataset_id = dataset_idx - num_sup_dataset
    out = model(inputs,set_id=real_dataset_id, mmen_grl=True)
    out_len, batch_size, _ = out.size()
    input_sizes = (input_sizes * out_len).long()
    prob = torch.softmax(out, dim=-1)
    loss = 0.1 * torch.sum(torch.mul(prob,torch.log2(prob+1e-8))) # H(x)
    return loss, out, batch_size, input_sizes,0,0

def code_from_last_version():
       # if dataset_idx < num_sup_dataset: # supervised
            #     inputs, input_sizes, targets, target_sizes, utt_list = data
            #     targets = targets.to(device)
            #     target_sizes = target_sizes.to(device)
            # else:
            #     inputs, input_sizes, utt_list = data
            #     real_dataset_id = dataset_idx - num_sup_dataset
                
            # if language_one_hot:
            #     B,T,_ = inputs.shape
            #     real_dataset_id = dataset_idx if dataset_idx < num_sup_dataset else (dataset_idx - num_sup_dataset)
            #     lids = torch.zeros((B,T,num_sup_dataset))
            #     lids[:,:,real_dataset_id] = 1
            #     inputs = torch.cat((inputs, lids), dim=-1)
            # inputs = inputs.to(device)
            # input_sizes = input_sizes.to(device)
            
            # if maml and dataset_idx < num_sup_dataset and is_training: # only support supervised training
            #     model_backup = copy.deepcopy(model) # gradient update on origin model.
            #     batch_size = inputs.shape[0]
            #     divide_index = int(batch_size/2)


            #     inputs_tr = inputs[:divide_index].clone().to(device)
            #     input_sizes_tr = input_sizes[:divide_index].clone().to(device)
            #     targets_tr = targets[:divide_index].clone().to(device)
            #     target_sizes_tr = target_sizes[:divide_index].clone().to(device)

            #     fake_out = model(inputs_tr, set_id=dataset_idx, mmen_grl=False)
            #     fake_out_len,_,_ = fake_out.size()
            #     fake_input_sizes = (input_sizes_tr * fake_out_len).long()
            #     fake_loss = loss_fn(fake_out, targets_tr, fake_input_sizes, target_sizes_tr)

            #     # get temp model theta'
            #     optimizer.zero_grad()

            #     fake_loss.backward()
            #     optimizer.step()

            #     # use whole test batch
            #     inputs = inputs[divide_index:]
            #     input_sizes = input_sizes[divide_index:]
            #     targets = targets[divide_index:]
            #     target_sizes = target_sizes[divide_index:]
            #     out = model(inputs, set_id=dataset_idx, mmen_grl=False)
            #     out_len, batch_size, _ = out.size()
            #     input_sizes = (input_sizes * out_len).long()
            #     loss = loss_fn(out, targets, input_sizes, target_sizes)

            #     optimizer.zero_grad()
            #     loss.backward()

            #     # restore theta' to theta
            #     model.load_state_dict(model_backup.state_dict())
            #     # theta + alpha * grad_with_theta'
            #     optimizer.step()

            # else:
            #     if dat:
            #         if dataset_idx < num_sup_dataset: # supervised
            #             out,language_out = model(inputs, set_id = dataset_idx)
            #         else:
            #             out,language_out = model(inputs, set_id = dataset_idx-num_sup_dataset)
            #         out_len, batch_size, _ = out.size()
            #         input_sizes = (input_sizes * out_len).long()

            #         language_target = torch.zeros(batch_size)

            #         if dataset_idx < num_sup_dataset: # supervised
            #             language_target = language_target.fill_(dataset_idx)
            #         else:
            #             language_target = language_target.fill_(dataset_idx - num_sup_dataset)
            #         # for i in range(batch_size):
            #         #     language_target[i][:input_sizes[i].item()] = dataset_idx
                    
            #         language_target = language_target.view(-1).long().to(device)
            #         ce = nn.CrossEntropyLoss(ignore_index=-1,reduction='sum')
            #         language_loss = ce(language_out, language_target)
                    
            #         if dataset_idx < num_sup_dataset: # supervised:
            #             asr_loss = loss_fn(out, targets, input_sizes, target_sizes)
            #         else:
            #             asr_loss = 0
            #         loss =  asr_loss+ language_loss
            #     else: # add min max
            #         if dataset_idx < num_sup_dataset:
            #             out = model(inputs, set_id=dataset_idx, mmen_grl=False)
            #             out_len, batch_size, _ = out.size()
            #             input_sizes = (input_sizes * out_len).long()
            #             loss = loss_fn(out, targets, input_sizes, target_sizes)
            #         else:
            #             if mme:
            #                 out = model(inputs,set_id=dataset_idx-num_sup_dataset, mmen_grl=True)
            #                 out_len, batch_size, _ = out.size()
            #                 input_sizes = (input_sizes * out_len).long()
            #                 prob = torch.softmax(out, dim=-1)
            #                 loss = 0.1 * torch.sum(torch.mul(prob,torch.log2(prob+1e-8))) # H(x)
            #             else:
            #                 continue


            # loss /= batch_size
            # cur_loss += loss.item()
            # total_loss += loss.item()
            # prob, index = torch.max(out, dim=-1)
            # if dataset_idx < num_sup_dataset:
            #     batch_errs, batch_tokens = model.compute_wer(index.transpose(0,1).cpu().numpy(), input_sizes.cpu().numpy(), targets.cpu().numpy(), target_sizes.cpu().numpy())
            # else:
            #     batch_errs = 0
            #     batch_tokens = 0 # do not compute wer

            #if not is_training:
            #    pdb.set_trace()
    pass

#dat=False, mme=False,language_one_hot=False, maml=False
def run_epoch(epoch_id, model, data_iter,loss_fn, device,opts,semi_data_iter=None, optimizer=None, print_every=20, is_training=True):
    if is_training:
        model.train()
    else:
        model.eval()
    
    total_loss = 0
    total_tokens = 0
    total_errs = 0
    cur_loss = 0
    num_sup_dataset = len(data_iter)
    if semi_data_iter:
        num_all_dataset = len(data_iter) + len(semi_data_iter)
        dataset_avail = [x for x in range(num_all_dataset)]
        generators = [gen_batch(loader) for loader in (data_iter + semi_data_iter)]
    else:
        dataset_avail = [x for x in range(len(data_iter))]
        num_all_dataset = len(data_iter)
        generators = [gen_batch(loader) for loader in data_iter]
    i = 0
    while len(dataset_avail) > 0:
        curr_idx = i % len(dataset_avail)
        dataset_idx = dataset_avail[curr_idx]
        curr_dataset = generators[dataset_idx]

        try:
            data = next(curr_dataset)
            supervised = True
            if not dataset_idx < num_sup_dataset:
                supervised = False
            forward_func = None
            if supervised:
                if opts.dat_lambda > 0: # dat
                    forward_func = dat_sup_training
                else:
                    forward_func = normal_sup_training
            else:
                if opts.dat_lambda > 0:
                    forward_func = dat_semi_training
                elif opts.mme_lambda > 0:
                    forward_func = mme_semi_training
                else:
                    continue # no semi-sup training
            loss, out, batch_size, input_sizes, targets, target_sizes = forward_func(model, loss_fn, data, device, dataset_idx, num_sup_dataset)
            loss /= batch_size
            cur_loss += loss.item()
            total_loss += loss.item()
            prob, index = torch.max(out, dim=-1)

            if supervised:
                batch_errs, batch_tokens = model.compute_wer(index.transpose(0,1).cpu().numpy(), input_sizes.cpu().numpy(), targets.cpu().numpy(), target_sizes.cpu().numpy())
            else:
                batch_errs = 0
                batch_tokens = 0 # do not compute wer

            total_errs += batch_errs
            total_tokens += batch_tokens

            if (i + 1) % print_every == 0 and is_training:
                print('Epoch = %d, step = %d, cur_loss = %.4f, total_loss = %.4f, total_cer = %.4f' % (epoch_id, 
                                     i+1, cur_loss / print_every, total_loss / (i+1), total_errs / total_tokens ))
                cur_loss = 0
        
        #not maml
            if is_training and True:    # maml do not update here
                optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_norm_(model.parameters(), 400)
                optimizer.step()

        except StopIteration:
            dataset_avail.remove(dataset_idx)
        
        i += 1
        
    average_loss = total_loss / (i+1)
    training = "Train" if is_training else "Valid"
    # if not is_training:
    #     pdb.set_trace()
    print("Epoch %d %s done, total_loss: %.4f, total_cer: %.4f" % (epoch_id, training, average_loss, total_errs / total_tokens))
    return 1-total_errs / total_tokens, average_loss

class Config(object):
    batch_size = 4
    dropout = 0.1

def main(conf):
    opts = Config()
    for k, v in conf.items():
        setattr(opts, k, v)
        print('{:50}:{}'.format(k, v))

    device = torch.device('cuda') if opts.use_gpu and torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    if opts.use_gpu:
        torch.cuda.manual_seed(opts.seed)
    datasets = os.listdir(opts.data_file)
    for idx,dataset in enumerate(datasets):
        datasets[idx] = opts.data_file + "/" + dataset
    train_scp = "/train/feats.scp"
    train_lab = "/train/lab.txt"
    valid_scp = "/dev/feats.scp"
    valid_lab = "/dev/lab.txt"
    vocab_f = "/units"
    if opts.universal:
        vocab_f = "/all_units"
    semi=False
    semi_loader=None

    #Data Loader
    vocab = [Vocab(dataset+vocab_f) for dataset in datasets]

    train_dataset = [SpeechDataset(voc, dataset+train_scp, dataset+train_lab, opts) for dataset,voc in zip(datasets,vocab)]
    dev_dataset = [SpeechDataset(voc, dataset+valid_scp, dataset+valid_lab, opts) for dataset,voc in zip(datasets,vocab)]
    train_loader = [SpeechDataLoader(dataset, batch_size=opts.batch_size, shuffle=opts.shuffle_train, num_workers=opts.num_workers) for dataset in train_dataset]
    dev_loader = [SpeechDataLoader(dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers) for dataset in dev_dataset]

    if opts.semi:
        semi=True
        semi_scp = "/train/feats_nolabel.scp"
        semi_train_dataset = [UnlabelSpeechDataset(dataset+semi_scp, opts) for dataset in datasets]
        semi_loader = [UnlabelSpeechDataLoader(dataset, batch_size=opts.batch_size, shuffle=opts.shuffle_train, num_workers=opts.num_workers) for dataset in semi_train_dataset]

    if opts.language_one_hot:
        # add size of one-hot label
        opts.rnn_input_size = opts.rnn_input_size + len(train_dataset)
    #Define Model
    rnn_type = supported_rnn[opts.rnn_type]
    rnn_param = {"rnn_input_size":opts.rnn_input_size, "rnn_hidden_size":opts.rnn_hidden_size, "rnn_layers":opts.rnn_layers, 
                    "rnn_type":rnn_type, "bidirectional":opts.bidirectional, "batch_norm":opts.batch_norm}
    
    num_class = [voc.n_words for voc in vocab]
    # opts.output_class_dim = vocab.n_words
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

    # Domain Adversarial Training
    if opts.dat_lambda != 0:
        dat = True
    else:
        dat = False

    if opts.mme_lambda != 0:
        mme = True
    else:
        mme = False
    model = Multi_CTC_Model(add_cnn=add_cnn, cnn_param=cnn_param, rnn_param=rnn_param, num_class=num_class, drop_out=drop_out,dat=opts.dat_lambda, mme=opts.mme_lambda,universal=opts.universal)
    model = model.to(device)
    num_params = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
    print("Number of parameters %d" % num_params)
    for idx, m in enumerate(model.children()):
        print(idx, m)
    
    if opts.resume != '' :
        print("Load ckp from {}".format(opts.resume))
        package = torch.load(opts.resume)
        state_dict = package['state_dict']
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model.state_dict().keys()}
        model_dict= model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


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
    loss_best = 1e6
    loss_best_true = 1e6
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
        
        train_acc, loss = run_epoch(count, model, train_loader,loss_fn, device,opts,semi_loader,optimizer=optimizer, print_every=opts.verbose_step, is_training=True)
        loss_results.append(loss)
        acc, dev_loss = run_epoch(count, model, dev_loader, loss_fn, device,opts, optimizer=None, print_every=opts.verbose_step, is_training=False)
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
