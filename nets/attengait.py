import tensorflow as tf
import tensorflow.keras.layers as layers
from losses.crossentropy_loss_all import CrossentropyAllLoss
from losses.triplet_loss_all import TripletBatchAllLoss
import os
from layers.matmul import MatMul
from layers.hpp import HPP
from layers.temporal_hpp import TemporalHPP
from layers.attention_conv import AttentionConv
from layers.attention_sum import AttentionHPP

class GaitSetTransformer():

	def __init__(self, experdir, reduce_channel, batch_size):
			self.model = None
			self.model_encode = None
			self.hist = None
			self.experdir = experdir
			self.reduce_channel = reduce_channel
			self.batch_size = batch_size
			self.losses = None
			self.losses_weights = None

	def load(self, previous_model, encode_layer='flatten'):
		self.model = tf.keras.models.load_model(previous_model,
								custom_objects={"TripletBatchAllLoss": TripletBatchAllLoss(), 'AttentionConv': AttentionConv()})
		self.model_encode= tf.keras.Model(self.model.input, self.model.get_layer(encode_layer).output)


	def build(self, optimizer, margin=0.2, input_shape=(30, 64, 44, 1), **kwargs):

		if 'weight_decay' in kwargs and kwargs['weight_decay'] > 0:
			regularizer = tf.keras.regularizers.L2(kwargs['weight_decay'])
		else:
			regularizer = None

		if 'attention_drop_rate' in kwargs:
			attention_drop_rate = kwargs['attention_drop_rate']
		else:
			attention_drop_rate = 0.25

		if 'crossentropy_weight' in kwargs:
			crossentropy_weight = kwargs['crossentropy_weight']
		else:
			crossentropy_weight = 0.0

		if 'nclasses' in kwargs:
			nclasses = kwargs['nclasses']
		else:
			nclasses = 50

		if 'norm_triplet_loss' in kwargs:
			norm_triplet_loss = kwargs['norm_triplet_loss']
		else:
			norm_triplet_loss = False
			
		if 'soft_triplet_loss' in kwargs:
			soft_triplet_loss = kwargs['soft_triplet_loss']
		else:
			soft_triplet_loss = False

		if 'split_crossentropy' in kwargs:
			split_crossentropy = kwargs['split_crossentropy']
		else:
			split_crossentropy = False

		if 'shared_weights_crossentropy' in kwargs:
			shared_weights_crossentropy = kwargs['shared_weights_crossentropy']
		else:
			shared_weights_crossentropy = False

		if 'n_filters' in kwargs:
			n_filters = kwargs['n_filters']
		else:
			n_filters = [32, 64, 128, 256]

		if 'softmax_attention' in kwargs:
			softmax_attention = kwargs['softmax_attention']
		else:
			softmax_attention = False

		if 'adaptative_loss' in kwargs:
			adaptative_loss = kwargs['adaptative_loss']
		else:
			adaptative_loss = False

		if 'reduction' in kwargs:
			reduction = kwargs['reduction']
		else:
			reduction = 'both'

		if 'combined_output_length' in kwargs:
			combined_output_length = kwargs['combined_output_length']
		else:
			combined_output_length = 16

		input_layer = layers.Input(shape=input_shape)

		# Initial conv
		model = layers.TimeDistributed(layers.Conv2D(n_filters[0], kernel_size=5, activation=tf.nn.swish, padding='same', use_bias=False,
											data_format='channels_last', kernel_regularizer=regularizer))(input_layer)

		# First conv block
		model = AttentionConv(3, regularizer=regularizer, dropout_rate=attention_drop_rate, softmax=softmax_attention)(model)
		model = layers.TimeDistributed(layers.Conv2D(n_filters[0], kernel_size=3, activation=tf.nn.swish, padding='same', use_bias=False, data_format='channels_last', kernel_regularizer=regularizer))(model)
		model = AttentionConv(3, regularizer=regularizer, dropout_rate=attention_drop_rate, softmax=softmax_attention)(model)
		model = layers.TimeDistributed(layers.Conv2D(n_filters[0], kernel_size=3, activation=tf.nn.swish, padding='same', use_bias=False, data_format='channels_last', kernel_regularizer=regularizer))(model)
		if nclasses > 999:
			model = AttentionConv(3, regularizer=regularizer, dropout_rate=attention_drop_rate, softmax=softmax_attention)(model)
			model = layers.TimeDistributed(layers.Conv2D(n_filters[0], kernel_size=3, activation=tf.nn.swish, padding='same', use_bias=False, data_format='channels_last', kernel_regularizer=regularizer))(model)
			model = AttentionConv(3, regularizer=regularizer, dropout_rate=attention_drop_rate, softmax=softmax_attention)(model)
			model = layers.TimeDistributed(layers.Conv2D(n_filters[0], kernel_size=3, activation=tf.nn.swish, padding='same', use_bias=False, data_format='channels_last', kernel_regularizer=regularizer))(model)

		model = layers.MaxPooling3D(pool_size=(1, 2, 2), data_format='channels_last')(model)

		# Second conv block
		model = AttentionConv(3, regularizer=regularizer, dropout_rate=attention_drop_rate, softmax=softmax_attention)(model)
		model = layers.TimeDistributed(layers.Conv2D(n_filters[1], kernel_size=3, activation=tf.nn.swish, padding='same', use_bias=False, data_format='channels_last', kernel_regularizer=regularizer))(model)
		model = AttentionConv(3, regularizer=regularizer, dropout_rate=attention_drop_rate, softmax=softmax_attention)(model)
		model = layers.TimeDistributed(layers.Conv2D(n_filters[1], kernel_size=3, activation=tf.nn.swish, padding='same', use_bias=False, data_format='channels_last', kernel_regularizer=regularizer))(model)
		if nclasses > 999:
			model = AttentionConv(3, regularizer=regularizer, dropout_rate=attention_drop_rate, softmax=softmax_attention)(model)
			model = layers.TimeDistributed(layers.Conv2D(n_filters[1], kernel_size=3, activation=tf.nn.swish, padding='same', use_bias=False, data_format='channels_last', kernel_regularizer=regularizer))(model)
			model = AttentionConv(3, regularizer=regularizer, dropout_rate=attention_drop_rate, softmax=softmax_attention)(model)
			model = layers.TimeDistributed(layers.Conv2D(n_filters[1], kernel_size=3, activation=tf.nn.swish, padding='same', use_bias=False, data_format='channels_last', kernel_regularizer=regularizer))(model)

		model = layers.MaxPooling3D(pool_size=(1, 2, 2), data_format='channels_last')(model)

		# Third conv block
		model = AttentionConv(3, regularizer=regularizer, dropout_rate=attention_drop_rate, softmax=softmax_attention)(model)
		model = layers.TimeDistributed(layers.Conv2D(n_filters[2], kernel_size=3, activation=tf.nn.swish, padding='same', use_bias=False, data_format='channels_last', kernel_regularizer=regularizer))(model)
		model = AttentionConv(3, regularizer=regularizer, dropout_rate=attention_drop_rate, softmax=softmax_attention)(model)
		model = layers.TimeDistributed(layers.Conv2D(n_filters[2], kernel_size=3, activation=tf.nn.swish, padding='same', use_bias=False, data_format='channels_last', kernel_regularizer=regularizer))(model)
		if nclasses > 999:
			model = AttentionConv(3, regularizer=regularizer, dropout_rate=attention_drop_rate, softmax=softmax_attention)(model)
			model = layers.TimeDistributed(layers.Conv2D(n_filters[2], kernel_size=3, activation=tf.nn.swish, padding='same', use_bias=False, data_format='channels_last', kernel_regularizer=regularizer))(model)
			model = AttentionConv(3, regularizer=regularizer, dropout_rate=attention_drop_rate, softmax=softmax_attention)(model)
			model = layers.TimeDistributed(layers.Conv2D(n_filters[2], kernel_size=3, activation=tf.nn.swish, padding='same', use_bias=False, data_format='channels_last', kernel_regularizer=regularizer))(model)

		# HPP
		model = AttentionConv(3, regularizer=regularizer, dropout_rate=attention_drop_rate, softmax=softmax_attention)(model)
		model_a = HPP(n_filters[2], regularizer=regularizer, activation_func=tf.nn.swish, reduction=reduction)(model)
		model_a = AttentionHPP(regularizer=regularizer, dropout_rate=attention_drop_rate, softmax=softmax_attention)(model_a)
		model_b = TemporalHPP(n_filters[2], regularizer=regularizer, activation_func=tf.nn.swish, reduction=reduction)(model)
		model_b = AttentionHPP(regularizer=regularizer, dropout_rate=attention_drop_rate, softmax=softmax_attention)(model_b)

		model = layers.Concatenate(axis=1)([model_a, model_b])
		model = layers.Lambda(lambda x: tf.transpose(x, [1, 0, 2]))(model)
		model = MatMul(bin_num=model.shape[0], hidden_dim=n_filters[3], input_dim=n_filters[2], regularizer=regularizer)(model)
		model = layers.Lambda(lambda x: tf.transpose(x, [1, 0, 2]))(model)
		model = AttentionHPP(regularizer=regularizer, dropout_rate=attention_drop_rate, softmax=softmax_attention)(model)
		model = layers.Lambda(lambda x: tf.transpose(x, [1, 0, 2]))(model)
		model = MatMul(bin_num=model.shape[0], hidden_dim=combined_output_length, input_dim=n_filters[3], regularizer=regularizer)(model)
		model = layers.Lambda(lambda x: tf.transpose(x, [1, 0, 2]))(model)
		model = AttentionHPP(regularizer=regularizer, dropout_rate=0.0, softmax=softmax_attention)(model)
		model = layers.Lambda(lambda x: tf.transpose(x, [1, 0, 2]), name="encode")(model)

		outputs = []
		outputs.append(model)
		losses = []
		losses.append(TripletBatchAllLoss(margin=margin, norm=norm_triplet_loss, soft=soft_triplet_loss, adaptative=adaptative_loss, n_parts=model.shape[0]))
		weights = []
		weights.append(1.0)
		if crossentropy_weight > 0.0:
			model2 = tf.transpose(model, [1, 0, 2])
			shapes_ = model2.get_shape().as_list()
			if split_crossentropy:
				activations = tf.split(model2, shapes_[1], axis=1)
				fc_list = []
				if shared_weights_crossentropy:
					fc_ = layers.Dense(nclasses, activation='softmax')
					for fc_ix in range(shapes_[1]):
						fc_list.append(fc_(activations[fc_ix]))
				else:
					for fc_ix in range(shapes_[1]):
						fc_ = layers.Dense(nclasses, activation='softmax')(activations[fc_ix])
						fc_list.append(fc_)

				combined_fcs = layers.Concatenate(axis=1, name='probs')(fc_list)
				outputs.append(combined_fcs)
				losses.append(CrossentropyAllLoss())
				weights.append(crossentropy_weight)
			else:
				model2 = tf.reshape(model2, [-1, shapes_[1] * shapes_[2]])
				model2 = layers.Dense(nclasses)(model2)
				model2 = layers.Softmax(name='probs')(model2)
				outputs.append(model2)
				losses.append(tf.keras.losses.SparseCategoricalCrossentropy())
				weights.append(crossentropy_weight)

		self.model = tf.keras.Model(inputs=input_layer, outputs=outputs)
		self.losses = losses
		self.losses_weights = weights
		self.model.compile(optimizer=optimizer, loss=losses, metrics=None, loss_weights=weights)
		self.model_encode = tf.keras.Model(inputs=input_layer, outputs=model)

	def fit(self, epochs, callbacks, training_generator, validation_generator, current_step=0, validation_steps=None, encode_layer=None, steps_per_epoch=None):
		self.hist = self.model.fit(training_generator, validation_data=validation_generator, epochs=epochs,
								   callbacks=callbacks, validation_steps=validation_steps, initial_epoch=current_step,
								   verbose=1, steps_per_epoch=steps_per_epoch)

		if encode_layer is None:
			out_layer = self.model.get_layer("code").output
		else:
			out_layer = self.model.get_layer(encode_layer).output

		self.model_encode = tf.keras.Model(self.model.input, out_layer)
		return len(self.hist.epoch)


	def predict(self, data, batch_size=128):
		pred = self.model.predict(data, batch_size=batch_size)
		return pred

	def encode(self, data, batch_size=128):
		features = self.model_encode.predict(data, batch_size=batch_size)

		features = tf.transpose(features, [1, 0, 2])
		shapes_ = tf.shape(features)
		features = tf.reshape(features, [-1, shapes_[1] * shapes_[2]])

		# Get the numpy matrix
		codes_norm = features.numpy()
		return codes_norm

	def save(self, epoch=None):
		if epoch is not None:
			# Save in such a way that can be recovered from different Python versions
			self.model.save_weights(os.path.join(self.experdir, "model-state-{:04d}_weights.hdf5".format(epoch)))
		else:
			# Save in such a way that can be recovered from different Python versions
			self.model.save_weights(os.path.join(self.experdir, "model-final_weights.hdf5"))


