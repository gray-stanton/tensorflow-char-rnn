import tensorflow as tf
import numpy as np
import os
from scipy.misc import logsumexp
#from tensorflow.framework

def _get_minimizer(options):
    return tf.train.AdagradOptimizer(learning_rate=options['learn-rate'])

def _make_rnn_layer(num_cells, name, options):
    #TODO: Add additional options
    act = tf.nn.relu
    if options['activation'] == 'RELU':
        act = tf.nn.relu
    return tf.contrib.rnn.GRUCell(
        num_cells,
        activation=act,
    )
def build_main_graph(params):
    """
    Construct the main tf graph for the char-rnn.

    Construct the main tf graph for the char-rnn, 
    based on a number of paramaters that can be set
    at model creation time.

    params (dict):
        Must contain keys 'seq-length', 'char-map-size',
        'label-length', 'hidden-layer-sizes', 'hidden-layer-details',
        'minimizer-options', 
    """
    with tf.name_scope('input'):
        char_input = tf.placeholder(tf.uint8,
                                    shape=[None, params['seq-length']],
                                    name='char_input')
        onehot_input = tf.one_hot(char_input, depth=params['char-map-size'],
                                 dtype=tf.float16, axis=2, name='one_hot_input')
        #Label to train against it the next character to come.
        char_label = tf.placeholder(tf.uint8,
                                    shape=[None, params['label-length']],
                                           name='char_label')
        onehot_label= tf.one_hot(char_label, depth=params['char-map-size'],
                                 dtype=tf.float16, axis=2,
                                 name='one_hot_label')
        init_hidden_states = []
        for i, shape in enumerate(params['hidden-layer-sizes']):
            new_placeholder = tf.placeholder(
                tf.float16,
                name='init_hidden_state_' + str(i),
                shape=[None, shape]
            )
            init_hidden_states.append(new_placeholder)
        init_hidden_states = tuple(init_hidden_states)

    
    hidden_nodes = []
    with tf.name_scope('hidden') as scope:
        for i, (size, layer_details) in enumerate(zip(params['hidden-layer-sizes'],
                                               params['hidden-layer-details'])):
            new_layer = _make_rnn_layer(size,
                                        name='rnn-layer-' + str(i),
                                        options=layer_details,
                                       )
            hidden_nodes.append(new_layer)
    
        multicell = tf.contrib.rnn.MultiRNNCell(hidden_nodes)
    with tf.name_scope('unroll'):
        #outputs:[batch-size, seq-length, hidden-layer-sizes[-1]]
        #final_states = (hidden-layer-sizes[0], hidden-layer-sizes[1], ...)
        print(init_hidden_states)
        print(multicell)
        outputs, final_states = tf.nn.dynamic_rnn(multicell,
                                                 onehot_input,
                                                 initial_state=init_hidden_states,
                                                 dtype=tf.float16,
                                                 time_major=False)
        final_state_stack = tf.stack(
            list(final_states), 
            axis = 0,
            name ='stack_states'
        )
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
        #long_output [batch-size * seq-length, hidden-layer-sizes[-1]]
        long_output = tf.reshape(outputs,
                                 [-1, params['hidden-layer-sizes'][-1]], 
                                    name='reshaped_output')
        long_logits = tf.matmul(long_output, logit_W) + logit_b
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

        global_step_tensor = tf.Variable(
            0, trainable = False, name='global_step'
        )
        minimizer = _get_minimizer(params['minimizer-options'])
        train_step = minimizer.minimize(mean_loss,  global_step_tensor)                              

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
            #rnn_cell_summaries = []
                #Use n.name[-2] to subtract off the ':0' that tf adds.
            #    rnn_cell_summaries.append(
            #        tf.summary.histogram(n.name[:-2] + 'summary', n)
            #    )
            input_labels_summary =tf.summary.tensor_summary(
                'input_labels_summary', char_label)
            # TODO: Add translation from chars to text 
            input_chars_summary = tf.summary.tensor_summary(
                'input_chars_summary', char_input)
        
        #Wrap up and hold state
        summaries = tf.summary.merge_all()
        tf.add_to_collection('input_placeholders', char_input)
        tf.add_to_collection('input_placeholders', char_label)
        for i in range(0, len(init_hidden_states)):
            tf.add_to_collection('input_placeholders', init_hidden_states[i])

        tf.add_to_collection('training_operations', train_step)
        tf.add_to_collection('summary_operations', summaries)
        tf.add_to_collection('summary_operations', mean_loss)
        tf.add_to_collection('summary_operations', global_step_tensor)
        tf.add_to_collection('states', final_state_stack)
        #tf.add_to_collection('states', init_hidden_states)
        tf.add_to_collection('outputs', char_probs)
        init_op = tf.global_variables_initializer()
        tf.add_to_collection('init_ops', init_op)


def create(model_dict):
    try:    
        model_dir = model_dict['model-dir']
        model_name = model_dict['model-name']
    except KeyError as e:
        print('Necessary configs not found!')
        print(model_dict)
        raise e
    metagraph_name = model_dir + model_name + '.meta'
    if metagraph_name in os.listdir(model_dir):
        raise ValueError('{} already contains a model!'.format(model_dir))
    graph = tf.Graph()
    with graph.as_default():
        build_main_graph(model_dict)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            #Run initialization, confirm graph build correctly
            init_op = tf.get_collection('init_ops')[0]
            sess.run([init_op])
            #Save graph for training
            saver.export_meta_graph(model_dir + model_name + '.meta')
            saver.save(sess, model_dir + model_name + '.ckpt',
                       write_meta_graph=False)






def get_init_state(layer_sizes, init, batch_size):
    """Initialize RNN state"""
    if init == 'ZERO':
        return [ np.zeros(dtype = np.float16, shape = (batch_size, s))
                for s in layer_sizes ]
        

def train(model_dict, batch_iterator):
    print(model_dict)
    try:
        model_name = model_dict['model-name']
        model_dir = model_dict['model-dir']
        layer_sizes = model_dict['hidden-layer-sizes']
        state_init = model_dict['state-init']
        print(layer_sizes)
    except KeyError as e:
        print('Necessary configs not found!')
        print(model_dict)
        raise e

    with tf.Session() as sess:
        #Load previously created graph
        try:
            new_saver = tf.train.import_meta_graph(
                model_dir + model_name + '.meta'
                )
        except FileNotFoundError as e:
            print('Can not find metagraph in {}'.format(model_dir))
            raise e

        ## Setup/Extract ops from graph
        new_saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        print('Restored!')
        #Initialize log writer for tensorboard
        train_writer = tf.summary.FileWriter(
            model_dir + '/train_logs', sess.graph
        )
        train_op = tf.get_collection('training_operations')[0]
        summary_op = tf.get_collection('summary_operations')[0]
        input_placeholders = tf.get_collection('input_placeholders')

        global_step_tensor = tf.get_collection('summary_operations')[2]
        step = global_step_tensor.eval()

        state_stack = tf.get_collection('states')[0]
        states = None
        #TRAIN LOOP
        for batch, label in batch_iterator:
            if states is None:
                states = get_init_state(
                    layer_sizes = layer_sizes,
                    init = state_init,
                    batch_size = batch.shape[0],
                    )


            feed_dict = {input_placeholders[0] : batch, # char_input
                         input_placeholders[1] : label, # char_label
                        }

            for i in range(2, len(layer_sizes) + 2):
                #State placeholders start at input_placeholders[2]
                feed_dict[input_placeholders[i]] = states[i-2]

            _, summary, state_stack_output = sess.run(
                [train_op, summary_op, state_stack], feed_dict
            )
            states = [state_stack_output[i] for  i in range(0, state_stack.shape[0])]
            step += 1
            if step % model_dict['report-freq'] == 0:
                print(step)
                train_writer.add_summary(summary, step)


        tf.assign(global_step_tensor, step)
        new_saver.save(sess, model_dir + model_name, global_step = step)


def evaluate():
    pass


def softmax_sample(probs, temperature=1.0):
    #Evaluate in log-scale to avoid numeric issues if T << 1 or >> 1
    lse = logsumexp(probs/temperature)
    log_scaled_probs = probs/temperature - lse
    scaled_probs = np.exp(log_scaled_probs)
    return np.random.choice(
        [i for i in range(0, len(scaled_probs))],
        p = scaled_probs)

                                                        

def generate(model_dict, seed_text):
    try: 
        model_name = model_dict['model-name']
        model_dir = model_dict['model-dir']
        layer_sizes = model_dict['hidden-layer-sizes']
        state_init = model_dict['state-init']
        char_map = model_dict['char-map']
        inv_char_map = model_dict['inv-char-map']
        seq_length = model_dict['seq-length']
        temp = model_dict['temperature']
        gen_length = model_dict['gen-length']
    except KeyError as e:
        print('Necessary configs not found!')
        print(model_dict)
        raise e
    with tf.Session() as sess:
        try:
            new_saver = tf.train.import_meta_graph(
                model_dir + model_name + '.meta'
                )
        except FileNotFoundError as e:
            print('Can not find metagraph in {}'.format(model_dir))
            raise e
        #Setup and extract necessary ops
        new_saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        char_probs = tf.get_collection('outputs')[0]
        state_stack_op = tf.get_collection('states')[0]
        input_placeholders = tf.get_collection('input_placeholders')

        #Pad with spaces up to expected sequence length
        pad_length = seq_length - len(seed_text)
        prompt = np.array([
            #Pad strings with ascii spaces mapped with char_map
            *[char_map[ord(' ')] for i in range(0, pad_length)],
            #Pad strings with ascii spaces mapped with char_map
            *[char_map[c] for c in seed_text]]).reshape(1,seq_length)
        states = None
        print(prompt)

        #Generate LOOP
        for i in range(0, gen_length):
            if states is None:
                states = get_init_state(
                    layer_sizes = layer_sizes,
                    init = state_init,
                    batch_size = 1,
                    )

                feed_dict = {
                    input_placeholders[0] : prompt,
                    input_placeholders[1] : (
                        np.zeros( (1, seq_length)).reshape(1, seq_length)),
                        }
            for i in range(2, len(layer_sizes)+ 2):
                #State placeholders start at input_placeholders[2]
                feed_dict[input_placeholders[i]] = states[i-2]

            probs, state_stack_output = sess.run([char_probs,
                                                  state_stack_op],
                                                 feed_dict)
            states = [state_stack_output[i]
                      for i in range(0, state_stack_op.shape[0])]
            last_char_probs = probs[0, seq_length - 1, :]
            #Sample with sofmax temp from char dist
            #then run through inv_char_map to gen next char
            next_char = chr(
                inv_char_map[ 
                    softmax_sample(last_char_probs, temp)]
            )

            if i <= pad_length:
                states = None # still in paded regime, reset state to 0
            old_text = ''.join([chr(inv_char_map[p])
                                for p in list(prompt[0, :])]
            )
            new_text = old_text[1:] + next_char
            prompt = np.array(
                [char_map[ord(c)] for c in new_text]).reshape(
                1, seq_length)
            print(new_text)


            





            



        























    
