#!/usr/bin/env python
"""char_rnn.py

Usage:
   char_rnn.py create <file> -sl <sl-length> --dir <model_dir> --param-file
   char_rnn.py train
   char_rnn.py generate
   char_rnn.py (-h | --help)

Options:
    -h --help   Prints usage
    --sequence-length -sl    Length of sequence


"""
from docopt import docopt
import tf_parser
import model
import os
if __name__ == '__main__':
    arguments = docopt(__doc__, version = 'char_rnn 0.1')
    print(arguments)
    if arguments['create']:
        try:
            with open(arguments['<file>'], 'r') as inputf:
                content = inputf.read()
                elem_length = int(arguments['<sl-length>'])
                model_dir = os.path.abspath(arguments['<model_dir>'])

                serial, char_map = tf_parser.translate(content)


        except Exception as e:
            raise e



