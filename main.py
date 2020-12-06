import tensorflow as tf
import numpy as np
from net import BiLSTMModel
from utils import *


def main():

	# read data
	train_tokens, train_tags = read_data('data/train.txt')
	validation_tokens, validation_tags = read_data('data/validation.txt')
	test_tokens, test_tags = read_data('data/test.txt')
	# create a dictionary
	special_tokens = ['<UNK>', '<PAD>']
	special_tags = ['O']

	# Create dictionaries
	token2idx, idx2token = build_dict(train_tokens + validation_tokens, special_tokens)
	tag2idx, idx2tag = build_dict(train_tags, special_tags)

	# train the model
	tf.reset_default_graph()

	model = BiLSTMModel(vocabulary_size=len(token2idx), n_tags=len(tag2idx), embedding_dim=200, n_hidden_rnn=200, PAD_index=token2idx['<PAD>'])

	batch_size = 32
	n_epochs = 4
	learning_rate = 0.005
	learning_rate_decay = np.sqrt(2)
	dropout_keep_probability = 0.5

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	print('Start training... \n')
	for epoch in range(n_epochs):
		# For each epoch evaluate the model on train and validation data
		print('-' * 20 + ' Epoch {} '.format(epoch+1) + 'of {} '.format(n_epochs) + '-' * 20)
		print('Train data evaluation:')
		eval_conll(model, sess, train_tokens, train_tags, token2idx, tag2idx, idx2token, idx2tag, short_report=True)
		print('Validation data evaluation:')
		eval_conll(model, sess, validation_tokens, validation_tags, token2idx, tag2idx, idx2token, idx2tag, short_report=True)

		# Train the model
		for x_batch, y_batch, lengths in batches_generator(batch_size, train_tokens, train_tags, token2idx, tag2idx, idx2token, idx2tag):
			model.train_on_batch(sess, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability)

		# Decaying the learning rate
		learning_rate = learning_rate / learning_rate_decay

	print('...training finished.')

	print('-' * 20 + ' Train set quality: ' + '-' * 20)
	train_results = eval_conll(model, sess, train_tokens, train_tags, token2idx, tag2idx, idx2token, idx2tag, short_report=False)

	print('-' * 20 + ' Validation set quality: ' + '-' * 20)
	validation_results = eval_conll(model, sess, validation_tokens, validation_tags, token2idx, tag2idx, idx2token, idx2tag, short_report=False)

	print('-' * 20 + ' Test set quality: ' + '-' * 20)
	test_results = eval_conll(model, sess, test_tokens, test_tags, token2idx, tag2idx, idx2token, idx2tag, short_report=False)

if __name__ == '__main__':
	main()