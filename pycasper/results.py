import os
import pdb
import json
import pandas as pd
import numpy as np
from warnings import warn

"""
walkthroughResults

  path: the directory where all the result files are stored
  args_subset: (optional) list of args to be added to the table; None if you want to add everything
  res_subset: (optional) list of result values to be added to the table; None if you want to add everything
  val_key: (optional) name of the result type to be used for selecting the best model
         must be present in the `result file` and `res_subset`
"""
def walkthroughResults(path, args_subset=None, 
                       res_subset=['train', 'val', 'test'], 
                       val_key='val'):
  ## discover args columns of a table
  for tup in os.walk(path):
    for fname in tup[2]:
      if fname.split('.')[-1] == 'args':
        all_args = json.load(open(os.path.join(tup[0], fname)))
        if args_subset is None:
          args_subset = all_args
        else:
          ## check if the args in the subset are available in the args
          for arg in args_subset:
            try:
              all_args[arg]
            except:
              warn('arg {} not in the args file of the model'.format(arg))
        ## assign [] to all args in args_subset
        best_df_dict = dict([(arg, []) for arg in args_subset])
        all_df_dict = dict([(arg, []) for arg in args_subset])
        
        break
    else:
      continue
    break

  ## discover result columns of the table
  for tup in os.walk(path):
    for fname in tup[2]:
      if fname.split('.')[-1] == 'json':
        all_res = json.load(open(os.path.join(tup[0], fname)))
        if res_subset is None:
          res_subset = all_res
        else:
          ## check if the res in the subset are available in the res.json file
          for res in res_subset:
            try:
              all_res[res]
            except:
              warn('res {} not in the res.json file of the model'.format(arg))
        assert np.array([r == val_key for r in res_subset]).any(), 'res_key not found in res_subset'
        ## assign [] to all res in res_subset
        best_df_dict.update(dict([(res, []) for res in res_subset]))
        all_df_dict.update(dict([(res, []) for res in res_subset]))
        break
    else:
      continue
    break
    
  ## add epoch to both the dictionaries
  best_df_dict.update({'epoch':[]})
  all_df_dict.update({'epoch':[]})

  ## add name to both dictionaries
  best_df_dict.update({'name':[]})
  all_df_dict.update({'name':[]})  
  
  for tup in os.walk(path):
    for fname in tup[2]:
      if fname.split('.')[-1] == 'json':
        ## load raw results
        res = json.load(open(os.path.join(tup[0],fname)))

        ## load args
        name = '_'.join(fname.split('.')[0].split('_')[:-1])
        args_path = '_'.join(fname.split('.')[0].split('_')[:-1] + ['args.args'])
        args = json.load(open(os.path.join(tup[0], args_path)))

        ## find the best result index
        min_index = np.argmin(res[val_key])

        ## add args to df_dict
        for arg in args_subset:
          best_df_dict[arg].append(args.get(arg))
          all_df_dict[arg].append(args.get(arg))

        ## add loss values to df_dict
        for r in res_subset:
          if res.get(r):
            best_df_dict[r].append(res.get(r)[min_index])
          else:
            best_df_dict[r].append(None)
          all_df_dict[r].append(res.get(r))

        ## add num_epochs to train to df_dict
        best_df_dict['epoch'].append(min_index+1)
        all_df_dict['epoch'].append(min_index+1)

        ## add name to dict
        best_df_dict['name'].append(name)
        all_df_dict['name'].append(name)

  ## Convert dictionary of results to a dataframe
  best_df = pd.DataFrame(best_df_dict)
  all_df = pd.DataFrame(all_df_dict)
  best_df = best_df[['name'] + list(args_subset) + ['epoch'] + list(res_subset)]
  all_df = all_df[['name'] + list(args_subset) + ['epoch'] + list(res_subset)]
  return best_df, all_df


def walkthroughModels(path):
  model_paths = []
  for tup in os.walk(path):
    for fname in tup[2]:
      if fname.split('_')[-1] == 'weights.p':
        model_paths.append(os.path.join(tup[0], fname))

  return model_paths
