import tensorflow as tf
import io
import os.path
from gensim.utils import deaccent
import re
import string
import numpy as np

def regularize_charset(fname):
    """Reduce the set of characters in the file to the minimum
    to encapsulate its semantics. Replaces non-ascii chars with their 
    ascii equivalent. Replaces non-printing chars with spaces, and tabs with 
    4 spaces.

    Arguments:
        fname: path to a text file to be encoded
    
    Returns:
        a file path with ascii chars replaced

    """
    with open(fname, 'r') as f:
        s = f.read()
        news = to_ascii(s)
        return write_with_suffix(fname, '-ascii')


def to_ascii(string):
    """
    Replace all non-ascii chars with ascii-equivalent, remove
    all non-printing characters,replace  all tabs with 4 spaces.
    
    Returns:
        A transformed string
    """
    tabs = re.compile('\t')
    newstring, _ = tabs.subn(' ' * 4, string)
    car_return_etc = re.compile('\r|\x0b|\x0c')
    newstring, _ = tabs.subn('\n', newstring)
    newstring = deaccent(newstring)
    nonprintable = re.compile('[^ -~]')
    newstring, _ = nonprintable.subn('', newstring)
    return newstring.encode('ascii')

def split_text(string, elem_length):
    """
    Splits a string into substrings of length elem_length, with
    space padding.

    Arguments:
        string: a string to split
        elem_length: length of substrings to split into

    Returns:
        A list of strings of length elem_length
    """
    rem = len(string) % elem_length
    padded_string = string + b' ' * rem

    #jDouble braces used to create a literal brace, re matches exactly
    # elem_length of any char 
    
    return [padded_string[i : i + elem_length]
            for i in range(0, len(padded_string) - elem_length, elem_length)]

def to_digit_array(string, char_map):
    """
    Convert a string into an nparray, mapping characters
    to ints based on char_map
    """
    return np.array([char_map[s] for s in string], dtype = np.int8)

def write_with_suffix(f, suffix):
    """Write a new txt file with the name of f concatted with the
    string suffix appended, with the same extension as the original file

    Arguments:
        f: a file object
        suffix: the suffix to be appended to the filename

    Returns:
        an file path to the writen file at fname + suffix.ext
    """
    fpath = f.name
    basename = os.path.basename(fpath)
    newname = basename + suffix

#rev_char_map = {i : c for i, c in char_map.items()}

def make_array(content, elem_length):
    """
    Take text string, process charset, create np array of dim [-1, elem_length]
    """
    ascii_content = to_ascii(content)
    charset = set(ascii_content)
    char_map = {c : i   for i, c in enumerate(sorted(list(charset)))}
    substrings = split_text(ascii_content, elem_length)
    array = np.array([to_digit_array(s, char_map) for s in substrings],
                     dtype = np.uint8)
    return array, char_map








