import pickle as pkl
import json
import os
from datetime import datetime
from tqdm import tqdm
import copy
import numpy as np
from pathlib import Path

from tensorboardX import SummaryWriter
import torch

from pycasper.name import Name

def accumulate_grads(model, grads_list):
  if grads_list:
    grads_list = [param.grad.data+old_grad.clone() for param, old_grad in zip(model.parameters(), grads_list)]
  else:
    grads_list += [param.grad.data for param in model.parameters()]
  return grads_list

def save_grads(val, file_path):
  pkl.dump(val, open(file_path, 'wb'))

def load_grads(file_path):
  return pkl.load(open(file_path))

class TensorboardWrapper():
  '''
  Wrapper to add values to tensorboard using a dictionary of values
  '''
  def __init__(self, log_dir):
    self.log_dir = log_dir

  def __call__(self, write_dict, comment='NA'):
    with SummaryWriter(log_dir=self.log_dir, comment=comment) as writer:
      for key in write_dict:
        for value in write_dict[key]:
          getattr(writer, 'add_' + key)(*value)

class BookKeeper():
  '''BookKeeper
  TODO: add documentation
  TODO: add save_optimizer_args as well
  TODO: choice of score kind to decide early-stopping (currently dev is default)
  Required properties in args
  - load
  - seed
  - save_dir
  - num_epochs
  - cuda
  - save_model
  - greedy_save
  - stop_thresh
  - eps
  - early stopping
  '''
  def __init__(self, args, args_subset,
               args_ext= 'args.args',
               name_ext='name.name',
               weights_ext = 'weights.p',
               res_ext='res.json',
               log_ext='log.log',
               args_dict_update = {},
               res = {'train':[], 'dev':[], 'test':[]},
               tensorboard = None):

    self.args = args
    self.args_subset = args_subset
    
    self.args_ext = args_ext.split('.')
    self.name_ext = name_ext.split('.')
    self.weights_ext = weights_ext.split('.')
    self.res_ext = res_ext.split('.')
    self.log_ext = log_ext.split('.')
    
    ## params for saving/notSaving models
    self.best_dev_score = np.inf
    self.stop_count = 0
    
    if self.args.load:
      if os.path.isfile(self.args.load):
        ## update the save_dir if the files have moved
        self.save_dir = '/'.join(args.load.split('/')[:-1])

        ## load Name
        self.name = self._load_name()
        
        ## load args
        self._load_args(args_dict_update)
        
      ## Serialize and save args
      self._save_args()

      ## load results
      self.res = self._load_res()

    else:
      self.save_dir = args.save_dir
      self.name = Name(self.args, *self.args_subset)

      ## save name
      self._save_name()

      ## Serialize and save args
      self._save_args()

      ## init empty results
      self.res = res

    ## Tensorboard 
    if tensorboard:
      self.tensorboard = TensorboardWrapper(log_dir=(Path(tensorboard)/Path(self.name.name+'tb')).as_posix())
    else:
      self.tensorboard = None

    ## seed numpy and torch
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
      
  def _load_name(self):
    name_filepath = '_'.join(self.args.load.split('_')[:-1] + ['.'.join(self.name_ext)])
    return pkl.load(open(name_filepath, 'rb'))

  def _save_name(self):
    name_filepath = self.name(self.name_ext[0], self.name_ext[1], self.save_dir)
    pkl.dump(self.name, open(name_filepath, 'wb'))

  def _load_res(self):
    print('Loading results')
    res_filepath = self.name(self.res_ext[0], self.res_ext[1], self.save_dir)
    return json.load(open(res_filepath))

  def _save_res(self):
    res_filepath = self.name(self.res_ext[0], self.res_ext[1], self.save_dir)
    json.dump(self.res, open(res_filepath,'w'))

  def update_res(self, res):
    for key in res:
      self.res[key].append(res[key])

  def update_tb(self, write_dict):
    if self.tensorboard:
      self.tensorboard(write_dict)
    else:
      warnings.warn('TensorboardWrapper not declared')

  def print_res(self, epoch, key_order=['train', 'dev', 'test'], exp=0):
    print_str = ', '.join(["exp: {}, epch: {}"] + ["{}: {}".format(key,{}) for key in key_order])
    result_list = [self.res[key][-1] for key in key_order]
    tqdm.write(print_str.format(exp, epoch, *result_list))

  def _load_args(self, args_dict_update):
    args_filepath = self.name(self.args_ext[0], self.args_ext[1], self.save_dir)
    if os.path.isfile(args_filepath):
      args_dict = json.load(open(args_filepath))
      ## any new argument to be updated
      args_dict.update(args_dict_update)

      ## update load path and cuda device to use
      args_dict.update({'load':self.args.load,
                        'cuda':self.args.cuda,
                        'save_dir':self.save_dir})

      self.args.__dict__.update(args_dict)

  def _save_args(self):
    args_filepath = self.name(self.args_ext[0], self.args_ext[1], self.save_dir)
    json.dump(self.args.__dict__, open(args_filepath, 'w'))

  def _load_model(self, model):
    weights_path = self.name(self.weights_ext[0], self.weights_ext[1], self.save_dir)
    model.load_state_dict(pkl.load(open(weights_path, 'rb')))
    
  def _save_model(self, model_state_dict):
    weights_path = self.name(self.weights_ext[0], self.weights_ext[1], self.save_dir)
    f = open(weights_path, 'wb') 
    pkl.dump(model_state_dict, f)
    f.close()

  def _copy_best_model(self, model):
    self.best_model = copy.deepcopy(model.state_dict())
    
  def _start_log(self):
    with open(self.name(self.log_ext[0],self.log_ext[1], self.save_dir), 'w') as f:
      f.write("S: {}\n".format(str(datetime.now())))
    
  def _stop_log(self):
    with open(self.name(self.log_ext[0],self.log_ext[1], self.save_dir), 'w') as f:
      f.write("E: {}\n".format(str(datetime.now())))

  def stop_training(self, model, epoch):
    ## copy the best model
    if self.res['dev'][-1]<self.best_dev_score:
      if self.args.greedy_save:
        save_flag = True
      else:
        save_flag=False
      self._copy_best_model(model)
      self.best_dev_score = self.res['dev'][-1]
    else:
      save_flag = False

    ## debug mode with no saving
    if not self.args.save_model:
      save_flag = False

    if save_flag:
      tqdm.write('Saving Model')
      self._save_model(self.best_model)

    ## early_stopping
    if self.args.early_stopping and len(self.res['train'])>=2:
      if (self.res['dev'][-2] - self.args.eps < self.res['dev'][-1]):
        self.stop_count += 1
      else:
        self.stop_count = 0

    if self.stop_count >= self.args.stop_thresh:
      print('Validation Loss is increasing')
      ## save the best model now
      if self.args.save_model:
        print('Saving Model by early stopping')
        pdb.set_trace()
        self._save_model(self.best_model)
      return True

    ## end of training loop
    if epoch == self.args.num_epochs-1 and self.args.save_model:
      print('Saving model after exceeding number of epochs')
      self._save_model(self.best_model)

    return False
