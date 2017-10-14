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
    arguments = docopt(__doc__, version = 'char_rnn 0.1')
    print(arguments)
    if arguments['create']:
        #Parse arguments into variables
        model_name = arguments['--model-name']
        model_dir  = os.path.abspath(arguments['--dir']) + '/'
        try:
            seq_length = int(arguments['--seq-length'])
            layer_sizes = [int(s) for s in arguments['--layers']]
            learn_rate = float(arguments['--learn-rate'])
        except Exception as e:
            print('Non-numeric argument!')
            raise e
        activations = [arguments['--activation']] * len(layer_sizes)
        
        #Open character input file and translate
        try:
            with open(arguments['<file>'], 'r') as inputf:
                content = inputf.read()
        except Exception as e:
            raise e
            
        serial, char_map = tf_parser.translate(content)
        inv_char_map = {v : k for k, v in char_map.items()}
        np.save(model_dir + model_name, serial)
        with open(model_dir + model_name + '.char_map', 'wb') as map_f:
            pickle.dump(char_map, map_f)
        with open(model_dir + model_name + '.inv_map', 'wb') as inv_f:
            pickle.dump(inv_char_map, inv_f)

        config = {
            'seq-length' : seq_length, 
            'char-map-size' : len(char_map),
            'label-length' : seq_length,
            'hidden-layer-sizes' : layer_sizes,
            'hidden-layer-details' : [{'activation' : a} for a in activations],
            'minimizer-options' : {'learning-rate' : learn_rate}
        }
        print(config)

        model.create(model_dir, config, model_name)

    if arguments['train']:
        model_name = arguments['--model-name']
        model_dir  = os.path.abspath(arguments['--dir']) + '/'
        try:
            seq_length = int(arguments['--seq-length'])
            layer_sizes = [int(s) for s in arguments['--shapes']]
            report_freq = int(arguments['--report-freq'])
            niter = int(arguments['<iterations>'])
            nepoch = int(arguments['--nepoch'])
            batch_size = int(arguments['--batch-size'])
        except Exception as e:
            print('Non-numeric argument!')
            raise e
        state_init = arguments['--state-init']


        #Load datafile
        try:
            serial = np.load(model_dir + model_name + '.npy')
        except Exception as e:
            print('Cannot find data file!')
            raise e

        batch_iterator = _make_batch_iterator(
            serial, seq_length, batch_size, niter, nepoch)

        config = {
            'report-freq' : report_freq,
            'state-init' : state_init,
            'shapes' : layer_sizes,
        }

        model.train(model_dir, config, batch_iterator, model_name)

    if arguments['generate']:
        model_name = arguments['--model-name']
        model_dir  = os.path.abspath(arguments['--dir']) + '/'
        try:
            seq_length = int(arguments['--seq-length'])
            layer_sizes = [int(s) for s in arguments['--shapes']]
            gen_length = int(arguments['--gen-length'])
            temp = float(arguments['--temp'])
        except Exception as e:
            print('Non-numeric argument!')
            raise e
        state_init = arguments['--state-init']
        seed_text  = arguments['<seed-text>']
        #Must be run through same transformation as raw text
        seed_text = tf_parser.to_ascii(seed_text)
        
        try:
            with open(model_dir + model_name + '.char_map', 'rb') as char_f:
                char_map = pickle.load(char_f)
            with open(model_dir + model_name + '.inv_map', 'rb') as inv_f:
                inv_map = pickle.load(inv_f)
        except Exception as e:
            print('Can not find char_map!')
            raise e

        config = {
            'seq-length' : seq_length,
            'char-map' : char_map,
            'inv-char-map' : inv_map,
            'state-init' : state_init,
            'temperature' : temp,
            'shapes' : layer_sizes
        }
        model.generate(model_dir, seed_text, gen_length, config, model_name)








