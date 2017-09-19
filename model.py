import tensorflow as tf
import numpy as np
import os
#from tensorflow.framework

def build_main_graph(params):
    init_op = tf.global_variables_initializer()
    with tf.name_scope('input'):
        char_input = tf.placeholder(tf.uint8,
                                    shape=[None, params['seq-length']],
                                    name='char_input')
        onehot_input = tf.one_hot(char_input, depth=params['char-map-size'],
                                 dtype=tf.float16, axis=-1, name='one_hot_input')
        #Label to train against it the next character to come.
        char_label = tf.placeholder(tf.uint8,
                                    shape=[None, params['label-length']],
                                           name='char_label')
        onehot_label= tf.one_hot(char_label, depth=params['char-map-size'],
                                 dtype=tf.float16, axis=-1,
                                 name='one_hot_label')
        init_hidden_states = []
        for i, size in enumerate(params['hidden-layer-sizes']):
            node = tf.placeholder(tf.float16, shape=[-1, size],
                                  name='init_hidden_' + str(i))
            init_hidden_states.append(node)
        # State unrolling uses tuples
        init_hidden_states = tuple(init_hidden_states)
    
    hidden_nodes = []
    with tf.name_scope('hidden') as scope:
        for i, (size, layer_details) in enumerate(params['hidden-layer-sizes'],
                                               params['hidden-layer-details']):
            hidden_nodes.append(make_rnn_layer(layer_details, size, scope))
    
        multicell = tf.contrib.rnn.MultiRNNCell(hidden_nodes)
        
        #outputs:[batch-size, seq-length, hidden-layer-sizes[-1]]
        #final_states = (hidden-layer-sizes[0], hidden-layer-sizes[1], ...)
        outputs, final_states = tf.nn.dynamic_rnn(multicell,
                                                 onehot_input,
                                                 initial_state=init_hidden_states,
                                                 dtype=tf.float16,
                                                 time_major=False)
    with tf.name_scope('output'):
        logit_W = tf.get_variable('logit_W',
                                shape=[params['hidden-layer-sizes'][-1],
                                       params['char-map-size']],
                                initializer=
                                  tf.contrib.layers.xavier_initializer(),
                                 dtype=tf.float16)
        logit_b = tf.get_variable('logit_b', 
                                  shape=params['char-map-size'],
                                  dtype=tf.float16)
        #long_output [batch-size * seq-length, char-map-size]
        long_output = tf.reshape(outputs, [-1, params['char-map-size']], 
                                    name='reshaped_output')
        long_logits = tf.matmul(long_output, logit_W) + b
        logits = tf.reshape(long_logits, [-1, params['seq-length'],
                                          params['char-map-size']])
        char_probs = tf.nn.softmax(logits)
        
        #If label-length isn't the full sequence, only backprop errors from
        #the last 'label-length' chars. Allows state burn-in.
        trimmed_logits = logits[:, -params['label-length']:, :]


        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=onehot_label,
            logits=trimmed_logits, dim=-1)
        mean_loss = tf.reduce_mean(losses)
        train_step = minimize(mean_loss, params['minimizer-options'])

        #Summary Nodes
        with tf.name_scope('summary'):
            mean_loss_summary = tf.summary.scalar('mean_loss_summary', mean_loss)
            losses_summary = tf.summary.histogram('losses_summary', losses)
            logits_summary = tf.summary.histogram('logits_summary', logits)

            #Weights
            logit_W_summary = tf.summary.histogram('logit_W_summary', logit_W)
            logit_W_serialized = tf.summary.tensor_summary('logit_W_serialized',
                                                           logit_W)
            logit_b_summary = tf.summary.histogram('logit_b_summary', logit_b)
            logit_b_serialized = tf.summary.tensor_summary('logit_b_serialized',
                                                           logit_b)
            rnn_cell_summaries = []
            for n in hidden_nodes:
                #Use n.name[-2] to subtract off the ':0' that tf adds.
                rnn_cell_summaries.append(
                    tf.summary.histogram(n.name[:-2] + 'summary', n)
                )
            input_labels_summary =tf.summary.tensor_summary(
                'input_labels_summary', char_label)
            # TODO: Add translation from chars to text 
            input_chars_summary = tf.summary.tensor_summary(
                'input_chars_summary', char_input)



def create(model_dir, params, model_name=None):
    if os.listdir(model_dir) != []:
        raise ValueError('New model directory not empty: {}'.format(model_dir))
    if model_name is None:
        model_name = 'my_model'
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.Saver()
        build_main_graph(params)
        summaries = tf.summary.merge_all()
        with tf.Session() as sess:
            # init_op is global_variables_initializer
            sess.run([init_op])
            saver.save(sess, model_name + '.ckpt')









def train():
    pass

def evaluate():
    pass




















    
