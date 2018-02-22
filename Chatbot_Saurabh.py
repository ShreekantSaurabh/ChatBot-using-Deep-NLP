# Building a ChatBot with Deep NLP

# Importing the libraries
import numpy as np
import tensorflow as tf     #For deep learning
import re     #Regular expression
import time


########## PART 1 - DATA PREPROCESSING ##########

# Importing the dataset
lines = open("movie_lines.txt", encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open("movie_conversations.txt", encoding = 'utf-8', errors = 'ignore').read().split('\n')

# Creating a dictionary that maps each line and its id (and remove all meta data)
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5: #each line in lines has 5 elements
        id2line[_line[0]] = _line[4]
        #_line[0] is line id and _line[4] is dialogue/statement in every line of lines

# Creating a list of all of the conversation's id or line's id (and remove all meta data)
conversations_ids = []
for conversation in conversations[:-1]:  #last row of conversations is empty so excluding it
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))

# Getting separately the questions (or 1st dialogue) and the answers (or 2nd dialogue which is replied to the 1st)
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)   #sub() is used for substitution or replacement
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

# Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
    
# Creating a dictionary that maps each word to its number of occurrences
#Will help in filtering out words that are rare (usually names,nouns that don't help the model learn). The less frequent words are removed with the below code.
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1   #word2count[word] making word as a key and 1 as a value of word2count dictionary
        else:
            word2count[word] += 1
            
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

# Creating two dictionaries that map the questions words and the answers words to a unique integer. Also Remove the non-frequent words.
threshold_questions = 20
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_questions:
        questionswords2int[word] = word_number
        #Every frequent word has been assigned by an unique number
        word_number += 1
        
threshold_answers = 20
answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_answers:
        answerswords2int[word] = word_number
        word_number += 1


# Adding the last tokens to these two dictionaries
#These tokens help the model understand the start and the end of a sentence.
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

# Creating the inverse dictionary of the answerswords2int dictionary
#The answersints2word helps form a sentence when the model predicts an output with integers.
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}

# Adding the End Of String token to the end of every answer
# Helps the model understand the end of each answer in the model.
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'


# Translating all the questions and the answers into integers ie. creating Bag of words and Replacing all the words that were filtered out by <OUT> 
#Converts every single word in questions and answers to integers and assigns the 'OUT' tag for the words that were filtered out in one of the above steps.
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
    
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)

# Sorting questions and answers by the length of questions to speedup the training and help to reduce the loss 
#(reduce amount of padding during training)
#Sorts questions and answers by their length. From 1 to 25.
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int):
        #i has index and question both. i[0] is index and i[1] is question
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])


########## PART 2 - BUILDING THE SEQ2SEQ MODEL ##########

#A seq2seq model is a type of many-to-many RNN model. They are most commonly used for chatbots and translation models. It has 2 components, the encoder and the decoder.

# Creating placeholders for the inputs and the targets
#A placeholder is simply a variable that we will assign data to at a later date. It allows us to create our operations and build our computation graph, without needing the data.
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')   #int32 is type of input data. [None, None] is shape of tensor ie. 2-dimension metrics as input data 
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')   #keep_probability is used to control drop-out rate of neurons.
    return inputs, targets, lr, keep_prob

# Preprocessing the targets  (target must be in batches each answer in a batch must start with <SOS> token)
#Assigns SOS tag to the start of every target(answers).
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])     #fill() creates a tensor of shape dimension ie. 2D [batch_size, 1] and fills it with word2int['<SOS>']
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])  #strided_slice() extracts subset of a tensor. Extracts target, begin at [0,0], end at [batch_size, -1], stride or slide size of [1,1] 
    preprocessed_targets = tf.concat([left_side, right_side], 1)  #concat() cancatenates tensors along 1-dimension. Concatenates [left_side, right_side] tensors along axis = 1 ie. horizontal cancatenation
    return preprocessed_targets

# Creating the Encoder RNN
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    #rnn_inputs = model_inputs(), rnn_size = no. of input tensor to rnn, sequence_length is list of the length of each question in a batch
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)    #Basic LSTM RNN cell
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)  #adds drop-out to the lstm. Operator adding dropout to inputs and outputs of the given cell.
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)    #RNN cell composed sequentially of multiple simple cells
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    #creates a dynamic version of bidirectional (forward & backward) RNN. The input size of forward & backward cell must match ie. cell_fw = encoder_cell, cell_bw = encoder_cell. Initial state for both direction is zero by default.
    #Making the encoder bidirectional proved to be much more effective than a simple feed forward network.
    return encoder_state
    #The Decoder RNN is built after decoding the training and test sets using the encoder state obtained from the above function

# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    #encoder_state is input to the decode_training_set. It is RNN cell in decoder layer. decoder_embedded_input is input on which we apply embedding before input. Embedding is a mapping from discrete objects, such as words, to vectors of real numbers. 
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])  #The model performs best when the attention states are set with zeros.
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size) #prepare function for attention mechanism.
    #The two attention options are bahdanau and luong. Bahdanau is less computationally expensive and better results were achieved with it.
    #Using attention in our decoding layers reduces the loss of our model by about 20% and increases the training time by about 20%. I’d say that it’s a fair trade-off.
    
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    #The attention_decoder_fn_train is a training function for an attention-based sequence-to-sequence model. It should be used when dynamic_rnn_decoder is in the training mode. The dynamic_rnn_decoder has two modes: training & inference.
    
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    #sequence_length is needed at training time, i.e., when inputs is not None, for dynamic unrolling. At test time, when inputs is None, sequence_length is not needed.

    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)


# Decoding the test/validation set
#code similar to decode_training_set()
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    #output_function of decode_training_set() is act as an input to the decode_test_set()
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    #The attention_decoder_fn_inference is a simple inference function for a sequence-to-sequence model. It should be used when dynamic_rnn_decoder is in the inference mode.
    #The extra parameters in attention_decoder_fn_inference (in compare to attention_decoder_fn_train) are necessary to help the model create accurate responses for your input sentences.
    #There is also no dropout in this function. This is because we are using it to create our responses during testing (aka making predictions), and we want to be using our full network for that.
    
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions

# Creating the Decoder RNN
#Code similar to encoder_rnn()
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    #We are using tf.variable_scope() to reuse the variables from training for making predictions.
    #Variable scope allows you to create new variables and to share already created ones while providing checks to not create or share by accident.
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        
        #Strongly encourage you to initialize your weights and biases. 
        #By initializing your weights with a truncated normal distribution and a small standard deviation, this can really help to improve the performance of your model.
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()

        #fully_connected Adds a fully connected last layer of LSTM. It creates a variable called weights, representing a fully connected weight matrix, which is multiplied by the inputs to produce a Tensor of hidden units.
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        
        #Here we are using the previous two functions, decode_training_set() & decode_test_set() to create our training_predictions and test_predictions. 
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions

# Building the seq2seq model
#Now that we have both the encoder RNN and the decoder RNN we'll use them to build our seq2seq model.
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    #embed_sequence() Maps a sequence of symbols to a sequence of embeddings. Typical use case would be reusing embeddings between an encoder and decoder.
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)   #input of the decoder_rnn() below.
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    #Similar to initializing weights and biases, I find it best to initialize embeddings as well. Rather than using a truncated normal distribution, a random uniform distribution is more appropriate.
    
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    #Outputs random values from a uniform distribution. The generated values follow a uniform distribution in the range [minval, maxval)
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    #Looks up ids in a list of embedding tensors. This function is used to perform parallel lookups on the list of tensors in parameters.
    
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions


########## PART 3 - TRAINING THE SEQ2SEQ MODEL ##########

# Setting the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

# Defining a session
tf.reset_default_graph()     #reset the graph before training. Clears the default graph stack and resets the global default graph.
session = tf.InteractiveSession()   #A TensorFlow Session for use in interactive contexts, such as a shell.
#interactive session provides a little more flexibility when building this model, but you can use whatever session type you wish.

# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')
#TensorFlow provides a placeholder operation that must be fed with data on execution.
#Sequence length will be the max line length for each batch. I sorted my inputs by length to reduce the amount of padding when creating the batches. This helped to speed up training.

# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)    #Returns the shape or dimension of a tensor.

# Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int)
#the input is often reversed in seq2seq models. This helps a model to produce better outputs because when the input data is being fed into the model, the start of the sequence will now become closer to the start of the output sequence.

# Setting up the Loss Error, the Optimizer (like any deep learning model) and Gradient Clipping.
#Gradient Clipping is a technique that will cap the gradients in graph between a min & max value to avoid the vanishing or exploding gradient issues.
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))    #Weighted cross-entropy loss for a sequence of logits. tf.ones([input_shape[0], sequence_length]) is weight.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]     #Clips tensor values to a specified min and max.
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
    #TensorFlow provides several operations that you can use to add clipping functions to your graph. 
    #You can use these functions to perform general data clipping, but they're particularly useful for handling exploding or vanishing gradients.

# Padding the sequences with the <PAD> token
#Padding are dummy values added to the sequence making the questions and answers to have the same length.
#Question: ['who','are','you',<PAD>,<PAD>,<PAD>,<PAD>]
#Answer:   [<SOS>,'I','am','a','bot','.',<EOS>,<PAD>]
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

# Splitting the data into batches of questions and answers since the RNN model accepts inputs only in batches.
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch

# Splitting the questions and answers into training and validation/test sets
#The model trains of 85% the data and tests its prediction on the 15% data to compute the loss error and improve in further epochs.
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]


# Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "./chatbot_weights.ckpt" # For Linux users, replace this line of code by: checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")


########## PART 4 - TESTING THE SEQ2SEQ MODEL ##########
  
# Loading the weights and Running the session
checkpoint = "./chatbot_weights.ckpt"   #During training, the weights are saved at every checkpoint. This can be loaded to chat with the chatbot.

#A TensorFlow Session for use in interactive contexts, such as a shell.The only difference with a regular Session is that an InteractiveSession installs itself as the default session on construction.
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)     #Saves and restores variables.The Saver class adds ops to save and restore variables to and from checkpoints. It also provides convenience methods to run these ops.
 
# Converting the questions from strings to lists of encoding integers
def convert_string2int(question, word2int):
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]
 
# Setting up the chat between chatbot and us.
while(True):
    question = input("You: ")
    if question == 'Goodbye':
        break
    question = convert_string2int(question, questionswords2int)
    question = question + [questionswords2int['<PAD>']] * (25 - len(question))
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answersints2word[i] == 'i':
            token = ' I'
        elif answersints2word[i] == '<EOS>':
            token = '.'
        elif answersints2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answersints2word[i]
        answer += token
        if token == '.':
            break
    print('ChatBot: ' + answer)
