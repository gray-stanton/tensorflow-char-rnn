#!/usr/bin/env python
"""char_rnn.py

Usage:
   char_rnn.py create (--dir=<model-dir>) (--layers=<layer-sizes> ...)
                      (--seq-length=<seq-length>) [--activation=<activation>]
                      [--learn-rate=<learn-rate>] [--model-name=<name>] <file>
   char_rnn.py train  (--dir=<model-dir>) (--seq-length=<seq-length>) 
                      (--shapes=<shapes> ...) (--batch-size=<batch-size>)
                      [--state-init=<state-init>] [--report-freq=<freq>]
                      [--model-name <name>]
                      [--nepoch=<nepoch>] <iterations>
   char_rnn.py generate (--dir=<model-dir>) (--seq-length=<seq-length>)
                        (--gen-length=<length>) (--shapes=<shapes> ...) 
                        [--state-init <state-init>] [--model-name <name>]
                        [--temp=<temp>] <seed-text>
   char_rnn.py (-h | --help)

Options:
    -h --help   Prints usage
    --dir <model-dir> -d <model-dir>  Model directory
    --layers <layer-sizes> ... -l <layer-sizes>...  RNN layer sizes
    --activation <activation>  RNN Activation function [default: RELU]
    --learn-rate <learn-rate>  Learning rate [default: 0.1]
    --seq-length <seq-length> -s <seq-length>  Truncated backprop length
    --model-name <name> -n <name>  Model name [default: my_model]
    --state-init <state-init>  Initialization type of RNN state [default: ZERO]
    --report-freq <freq> -f <freq>  Report frequency [default: 1]
    --seed-text <seed> -x  Seed text for generation
    --gen-length <length> -g <length>  Number of characters to generate
    --shapes <shapes>  Model layer sizes FIX
    --temp <temp> -t <temp>  Softmax temperature for gen [default: 1]
    --nepoch <nepoch> -e  Number of epochs [default: 1]
    --batch-size <batch-size> -b <batch-size>  Minibatch size 

"""
from docopt import docopt
import tf_parser
import model
import os
import pickle
import numpy as np
import math
from yaml import load, dump

def _arg_parse(docopt_dict, model_dict = None):
    """ 
    Extract entries from docopt dict, return correctly-typed model config
    dict. If model_dict already exists, update/overwrite.

    """
    if model_dict == None:
        model_dict = {}

    for k, v in docopt_dict.items():
        nkey, nval = _parse(k, v)
        if nkey in model_dict:
            print('Overwriting {} with {} from {}'.format(nkey, nval, v))
        model_dict[nkey] = nval
    return model_dict


def _pack(model_dict):
    model_dict['minimizer-options'] = {
        'learn-rate' : model_dict['learn-rate']
    }
    del model_dict['learn-rate']
    n_layers = len(model_dict['hidden-layer-sizes'])
    model_dict['hidden-layer-details'] = [
        {'activation' : a} for a in range(1, n_layers)]
    del model_dict['activation']

    return model_dict

def _parse(key, value):
    """Parses a key value pair into the correct type
    Handles: '--model-name', '--dir', '--layers', '--activation'
       '--state-init', '--report-freq', '--seedtext', '--gen-length',
       '--shapes', '--temp', '--nepoch', '--batch-size'
    """

    def _nonneg_check(x):
        if x < 0:
            raise ValueError(
                'Numeric config arguments must be >=0!'
            )
    if value is None:
        return key, value
    elif key == '--dir':
        nkey = 'model-dir'
        nvalue = os.path.abspath(value) + '/' 
    elif key == '--model-name':
        nkey = 'model-name'
        nvalue = value
    elif key == '--seq-length':
        nkey = 'seq-length'
        nvalue = int(value)
        _nonneg_check(nvalue)
    elif key == '--learn-rate':
        nkey = 'learn-rate'
        nvalue = float(value)
        _nonneg_check(nvalue)
    elif key == '--activation':
        nkey = 'activation'
        nvalue = value
    elif key == '--layers':
        nkey = 'hidden-layer-sizes'
        nvalue = [int(v) for v in value]
        [_nonneg_check(x) for x in nvalue]
    elif key == '--report-freq':
        nkey = 'report-freq'
        nvalue = int(value)
        _nonneg_check(nvalue)
    elif key == '--state-init':
        nkey = 'state-init'
        nvalue = value
    elif key == '<iterations>':
        nkey = 'train-iter'
        nvalue = int(value)
        _nonneg_check(nvalue)
    elif key == '--nepoch':
        nkey = 'train-epochs'
        nvalue = int(value)
        _nonneg_check(nvalue)
    elif key == '--batch-size':
        nkey = 'batch-size'
        nvalue = int(value)
        _nonneg_check(nvalue)
    elif key == '--gen-length':
        nkey = 'gen-length'
        nvalue = int(value)
        _nonneg_check(nvalue)
    elif key == '--temp':
        nkey = 'temperature'
        nvalue = float(value)
        _nonneg_check(nvalue)
    elif key == '<file>' :
        nkey = 'raw-file'
        nvalue = os.path.abspath(value)
    elif key == '<seed-text>':
        nkey = 'seed'
        nvalue = value
    else:
        nkey = key
        nvalue = value
        print('Unknown key: {}'.format(key))
    
    return nkey, nvalue





def _make_batch_iterator(arr, length, batch_size, niter, nepoch):
    for e in range(0, nepoch):
        n_batches_per_epoch = math.floor(len(arr)/(batch_size * length))
        if niter > n_batches_per_epoch:
            print('Too many iterations per epoch!')

        start_points = np.array(
            range(0, len(arr), n_batches_per_epoch * length)[0:batch_size])

        def get_batch(start_points):
            indexes = np.array([np.arange(s, s+length) for s in list(start_points)])
            inputs = arr[indexes]
            labels = arr[indexes+1]
            return inputs, labels

        for i in range(0, niter):
            batch = get_batch(start_points)
            start_points = start_points + length
            yield batch





if __name__ == '__main__':
    docopt_dict= docopt(__doc__, version = 'char_rnn 0.1')
    print(docopt_dict)
    if docopt_dict['create']:
        #Parse arguments into variables
        model_dict = _arg_parse(docopt_dict, None)
        model_dict = _pack(model_dict)
        seq_length = model_dict['seq-length']
        #Open character input file and translate
        try:
            with open(model_dict['raw-file'], 'r') as inputf:
                content = inputf.read()
        except (IOError, FileNotFoundError) as e:
            raise e
            
        serial, char_map = tf_parser.translate(content)
        inv_char_map = {v : k for k, v in char_map.items()}
        model_dict['char-map'] = char_map
        model_dict['inv-char-map'] = inv_char_map
        model_dict['char-map-size'] = len(char_map)
        #TODO: Add config options for label-length < seq-length
        model_dict['label-length'] = seq_length


        #Build graph and save metagraph etc.
        model.create(model_dict)
       
       #Save data and configuration
        model_dict['data-file'] = model_dir + model_name + '.npy'
        model_dict['config-file'] = model_dir + model_name + '_config.yaml'

        np.save(model_dict['data-file'], serial)
        with open(model_dict['config-file'], 'w') as yamlfile:
            yaml.dump(model_dict, yamlfile)

    if docopt_dict['train']:
        model_name = docopt_dict['--model-name']
        model_dir  = os.path.abspath(docopt_dict['--dir']) + '/'
        config_file = model_dir + model_name + '_config.yaml'
        try:
            with open(config_file, 'r') as yamlfile:
                old_model_dict = yaml.load(yamlfile)
        except (IOError, FileNotFoundError) as e:
            print('Cannot find {}'.format(config_file))
            raise e
        model_dict = _arg_parse(docopt_dict, old_model_dict)

        #Load datafile
        try:
            serial = np.load(model_dict['data-file'])
        except (IOError, FileNotFoundError) as e:
            print('Cannot find data file {}!'.format(model_dict['data-file']))
            raise e

        #Create iterator for number of batches
        #TODO: Test to see if should switch to QueueRunners
        batch_iterator = _make_batch_iterator(
            serial,
            model_dict['seq-length'],
            model_dict['batch_size'],
            model_dict['train-iter'],
            model_dict['train-epochs'])


        #Run model training loop
        model.train(model_dict, batch_iterator)

        #Update configuration file
        with open(model_dict['config-file'], 'w') as yamlfile:
            yaml.dump(model_dict, yamlfile)

    if docopt_dict['generate']:
        model_name = docopt_dict['--model-name']
        model_dir  = os.path.abspath(docopt_dict['--dir']) + '/'
        config_file = model_dir + model_name + '_config.yaml'
        try:
            with open(config_file, 'r') as yamlfile:
                old_model_dict = yaml.load(yamlfile)
        except (IOError, FileNotFoundError) as e:
            print('Cannot find {}'.format(config_file))
            raise e
        model_dict = _arg_parse(docopt_dict, old_model_dict)

        #Must be run through same transformation as raw text
        seed_text = tf_parser.to_ascii(model_dict['seed'])
        

        model.generate(model_dict, seed_text)

        #Update configuration file
        with open(model_dict['config-file'], 'w') as yamlfile:
            yaml.dump(model_dict, yamlfile)







