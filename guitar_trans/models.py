from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from builtins import zip
from builtins import range
from builtins import object
from past.utils import old_div
import os, sys, time, random
import numpy as np
import librosa as rosa
import theano
import theano.tensor as T
import lasagne
import pprint
from lasagne import layers
from sklearn.metrics import confusion_matrix
from .parameters import MC_LENGTH, SAMPLING_RATE, HOP_LENGTH

#===== FUNCTIONS =====#

def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def categorical_crossentropy_logdomain(log_predictions, targets):
    return -T.sum(targets * log_predictions, axis=1)

#===== FUNCTIONS =====#

class Feature(object):
    @staticmethod
    def extract_features(y, mc, fn, ans=None):
        # MUST BE OVERRIDDEN
        return None

    @staticmethod
    def melody_features(mc, norm=True):
        nmc = old_div((mc - np.mean(mc)), np.std(mc)) if norm else mc # normalize melody contour
        dmc = np.gradient(nmc) # calculate the gradient (first derivative) of melody contour
        return nmc, dmc

class RawFeature(Feature):
    @staticmethod
    def extract_features(y, mc, fn, ans=None):
        nmc, dmc = Feature.melody_features(mc)
        if np.any(np.isnan([nmc, dmc])):
            print('nan in {}.'.format(fn))
            return None
        ra = np.reshape(y, (1,-1))
        # (raw_audio, melody_contour, 1st derivative of mc, answer, file_name) for each element
        mc_all = np.reshape([mc, dmc], (2,-1)) 
        return (ra, mc_all, fn) if ans is None else (ra, mc_all, ans, fn)

class MFCCFeature(Feature):
    @staticmethod
    def extract_features(y, mc, fn, ans=None):
        nmc, dmc = Feature.melody_features(mc)
        if np.any(np.isnan([nmc, dmc])):
            print('nan in {}.'.format(fn))
            print(mc)
            return None
        n_mfcc = 13
        mfcc = rosa.feature.mfcc(y, sr=SAMPLING_RATE, n_mfcc=n_mfcc, n_fft=512, hop_length=HOP_LENGTH)
        mfcc_d = rosa.feature.delta(mfcc)
        mfcc_d2 = rosa.feature.delta(mfcc, order=2)
        # feat_all = np.concatenate((mfcc, mfcc_d, mfcc_d2), axis=0).astype('float32')
        feat_all = np.concatenate((mfcc, mfcc_d, mfcc_d2, np.array([nmc]), np.array([dmc])), axis=0).astype('float32')
        return (feat_all, fn) if ans is None else (feat_all, ans, fn)

class SpecFeature(Feature):
    @staticmethod
    def extract_features(y, mc, fn, ans=None):
        nmc, dmc = Feature.melody_features(mc)
        if np.any(np.isnan([nmc, dmc])):
            print('nan in {}.'.format(fn))
            return None
        n_mels = 128
        melspec = rosa.feature.melspectrogram(y, sr=SAMPLING_RATE, n_fft=512, hop_length=HOP_LENGTH, n_mels=n_mels)
        feat_all = np.concatenate((melspec, np.array([mc]), np.array([dmc])), axis=0).astype('float32')
        return (feat_all, fn) if ans is None else (feat_all, ans, fn)

class CocktailFeature(Feature):
    @staticmethod
    def extract_features(y, mc, fn, ans=None):
        nmc, dmc = Feature.melody_features(mc)
        if np.any(np.isnan([nmc, dmc])):
            print('nan in {}.'.format(fn))
            return None
        n_mels = 128
        n_mfcc = 13
        mfcc = rosa.feature.mfcc(y, sr=SAMPLING_RATE, n_mfcc=n_mfcc, n_fft=512, hop_length=HOP_LENGTH)
        mfcc_d = rosa.feature.delta(mfcc)
        mfcc_d2 = rosa.feature.delta(mfcc, order=2)
        melspec = rosa.feature.melspectrogram(y, sr=SAMPLING_RATE, n_fft=512, hop_length=HOP_LENGTH, n_mels=n_mels)
        feat_all = np.concatenate((mfcc, mfcc_d, mfcc_d2, melspec, np.array([nmc]), np.array([dmc])), axis=0).astype('float32')
        return (feat_all, fn) if ans is None else (feat_all, ans, fn)

#===== MODELS =====#

class Model(object):
    def __init__(self, net_opts, fp):
        self.net_opts = net_opts
        self.fp = fp
        self.init_model()

    #===== LAYERS =====#
    def set_conv_layer(self, network, layer_name, dropout=True, pad=0, bnorm=False):
        opts = self.net_opts[layer_name]
        ll = layers.Conv1DLayer(
                layers.dropout(network, p=self.net_opts['dropout_p']) if dropout else network,
                num_filters=opts['num_filters'],
                filter_size=opts['filter_size'],
                stride=opts['stride'],
                pad=pad,
                name=layer_name
             )
        return layers.batch_norm(ll) if bnorm else ll

    def set_pool_layer(self, network, layer_name):
        opts = self.net_opts[layer_name]
        return layers.Pool1DLayer(network, opts['pool_size'], 
                                  mode=opts['mode'], 
                                  ignore_border=False
                                 )

    def iterate_minibatches(self, inputs, batchsize):
        for start_idx in range(0, len(inputs), batchsize):
            yield inputs[start_idx:start_idx + batchsize]

    def init_model(self):
        # MUST BE OVERRIDDEN
        self.build_network()
        self.train_fn = None
        self.val_fn = None
        self.run_fn = None

    def build_network(self):
        # MUST BE OVERRIDDEN
        self.network = None
        return self.network

    def train(self, feature_list, num_epochs=60):
        print('Start training...')
        sys.stdout.flush()
        np.random.shuffle(feature_list)
        ch = old_div(len(feature_list), 5)
        val_list, train_list = feature_list[:ch], feature_list[ch:]
        lowest_loss = 100.0
        temp_model_file = '.temp_{}'.format(os.path.basename(self.fp))
        temp_model_fp = os.path.join(os.path.dirname(self.fp), temp_model_file)
        for epoch in range(num_epochs):
            start_time = time.time()
            train_err, train_batches = self.train_one(train_list)
            val_err, val_acc, val_batches = self.val_one(val_list)
            
            # Print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(old_div(train_err, train_batches)))
            val_loss = old_div(val_err, val_batches)
            print("  validation loss:\t\t{:.6f}".format(val_loss))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

            # Save the lowest loss
            if val_loss < lowest_loss:
                lowest_loss = val_loss
                final_param = lasagne.layers.get_all_param_values(self.network)
                self.save(temp_model_fp, final_param)

        os.rename(temp_model_fp, self.fp)

    def test(self, feature_list):
        # MUST BE OVERRIDDEN
        return None

    def run(self, feature_list):
        pred_list = []
        for bt in self.iterate_minibatches(feature_list, 10):
            feat, fn = list(zip(*bt))
            pred = self.run_fn(list(feat))
            pred_list += pred[0].tolist()
        pred_list = np.array(pred_list)
        return pred_list

    def set_param_values(self, val):
        lasagne.layers.set_all_param_values(self.network, val)

    def train_one(self, train_list):
        # MUST BE OVERRIDDEN
        train_err = 0
        train_batches = 1
        print('Do nothing in Model class! Must override this function!')
        return train_err, train_batches

    def val_one(self, val_list):
        # MUST BE OVERRIDDEN
        val_err = 0
        val_acc = 0
        val_batches = 1
        print('Do nothing in Model class! Must override this function!')
        return val_err, val_acc, val_batches 

    def save(self, save_fp=None, params=None):
        if save_fp is None:
            save_fp = self.fp
        if params is None:
            params = lasagne.layers.get_all_param_values(self.network)
        np.savez(save_fp, net_opts=self.net_opts, 
                          class_name=self.__class__.__name__.split('.')[-1],
                          params=params)

    @staticmethod
    def init_from_file(model_fp):
        npzfile = np.load(model_fp,encoding="latin1")
        print(npzfile['class_name'])
        
        # print("======= npzfile.files =======")
        print(npzfile.files)

        # Handle model string in python3 
        class_name = npzfile['class_name'].item()
        class_name = class_name.decode("utf-8")
        model_class = globals()[class_name]

        # Handle model string in python2
        # model_class = globals()[npzfile['class_name'].item()]


        # print("======= npzfile.net_opts =======")
        # print(npzfile['net_opts'].item())
        model = model_class(npzfile['net_opts'].item(), model_fp)
        
        # print("======= npzfile.params =======")
        # print(npzfile['params'])
        model.set_param_values(npzfile['params'])
        
        return model

##### MLP Network
class DNNModel(Model):
    def init_model(self):
        print('Initializing model...')
        mfcc_input_var = T.tensor3('mfcc_input')
        target_var = T.imatrix('targets')
        network = self.build_network(mfcc_input_var)
        prediction = layers.get_output(network)
        prediction = T.clip(prediction, 1e-7, 1.0 - 1e-7)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        params = layers.get_all_params(network, trainable=True)
        # updates = lasagne.updates.adagrad(loss, params, learning_rate=0.002)
        updates = lasagne.updates.sgd(loss, params, learning_rate=0.02)

        test_prediction = layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)),
                          dtype=theano.config.floatX)

        print('Building functions...')
        self.train_fn = theano.function([mfcc_input_var, target_var], 
                                        [loss, prediction], 
                                        updates=updates, 
                                        on_unused_input='ignore')
        self.val_fn = theano.function([mfcc_input_var, target_var], 
                                        [test_loss, test_acc, test_prediction], 
                                        on_unused_input='ignore')
        self.run_fn = theano.function([mfcc_input_var],
                                        [prediction],
                                        on_unused_input='ignore')

    def test(self, feature_list):
        test_err = 0
        test_acc = 0
        test_batches = 0
        ans_list, pred_list = [], []
        for bt in self.iterate_minibatches(feature_list, 10):
            feat, ans, fn = list(zip(*bt))
            err, acc, pred = self.val_fn(list(feat), list(ans))
            test_err += err
            test_acc += acc
            test_batches += 1
            for a, p in zip(ans, pred):
                ans_list.append(np.argmax(a))
                pred_list.append(np.argmax(p))
        confusion_mat = confusion_matrix(ans_list, pred_list)
        print("  test loss:\t\t{:.6f}".format(old_div(test_err, test_batches)))
        print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
        print('confusion matrix:')
        print(confusion_mat)
        return confusion_mat

    def train_one(self, train_list):
        train_err = 0
        train_batches = 0
        for bt in self.iterate_minibatches(train_list, 10):
            feat, ans, fn = list(zip(*bt))
            err, pred = self.train_fn(list(feat), list(ans))
            # if random.randint(0, 19) == 0:
            #   print 'err', err
            #   print 'pred', pred
            train_err += err
            train_batches += 1
        return train_err, train_batches

    def val_one(self, val_list):
        val_err = 0
        val_acc = 0
        val_batches = 0
        for bt in self.iterate_minibatches(val_list, 10):
            feat, ans, fn = list(zip(*bt))
            err, acc, pred = self.val_fn(list(feat), list(ans))
            val_err += err
            val_acc += acc
            val_batches += 1
        return val_err, val_acc, val_batches 

class MFCCDNNModel(DNNModel, MFCCFeature):
    def build_network(self, mfcc_input_var):
        print('Building dnn with parameters:')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.net_opts)

        mfcc_network = layers.InputLayer((None, 41, MC_LENGTH), mfcc_input_var)
        mfcc_network = layers.BatchNormLayer(mfcc_network)
        for n in self.net_opts['layer_list']:
            mfcc_network = layers.DenseLayer(layers.dropout(mfcc_network, p=self.net_opts['dropout_p']), 
                                            n, 
                                            nonlinearity=lasagne.nonlinearities.rectify)
        mfcc_network = layers.DenseLayer(layers.dropout(mfcc_network, p=self.net_opts['dropout_p']), 
                                        self.net_opts['num_class'], 
                                        nonlinearity=lasagne.nonlinearities.softmax)
        
        self.network = mfcc_network
        return self.network

class SpecDNNModel(DNNModel, SpecFeature):
    def build_network(self, mspec_input_var):
        print('Building spec dnn with parameters:')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.net_opts)

        mspec_network = layers.InputLayer((None, 130, MC_LENGTH), mspec_input_var)
        mspec_network = layers.BatchNormLayer(mspec_network)
        for n in self.net_opts['layer_list']:
            mspec_network = layers.DenseLayer(layers.dropout(mspec_network, p=self.net_opts['dropout_p']), 
                                            n, 
                                            nonlinearity=lasagne.nonlinearities.rectify)
        mspec_network = layers.DenseLayer(layers.dropout(mspec_network, p=self.net_opts['dropout_p']), 
                                        self.net_opts['num_class'], 
                                        nonlinearity=lasagne.nonlinearities.softmax)
        
        self.network = mspec_network
        return self.network

##### CNN
class MFCCCNNModel(DNNModel, MFCCFeature):
    def build_network(self, mfcc_input_var):
        print('Building cnn with parameters:')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.net_opts)

        mfcc_network = layers.InputLayer((None, 41, MC_LENGTH), mfcc_input_var)
        mfcc_network = layers.BatchNormLayer(mfcc_network)
        mfcc_network = self.set_conv_layer(mfcc_network, 'conv_1', bnorm=False)
        mfcc_network = self.set_pool_layer(mfcc_network, 'pool_1')
        mfcc_network = self.set_conv_layer(mfcc_network, 'conv_2', bnorm=False)
        mfcc_network = self.set_pool_layer(mfcc_network, 'pool_2')
        for n in self.net_opts['layer_list']:
            # mfcc_network = layers.batch_norm(layers.DenseLayer(layers.dropout(mfcc_network, p=self.net_opts['dropout_p']), 
            #                                  n, 
            #                                  nonlinearity=lasagne.nonlinearities.rectify)
            #                                 )
            mfcc_network = layers.DenseLayer(layers.dropout(mfcc_network, p=self.net_opts['dropout_p']), 
                                            n, 
                                            nonlinearity=lasagne.nonlinearities.rectify)
            # mfcc_network = layers.BatchNormLayer(mfcc_network)
        mfcc_network = layers.DenseLayer(layers.dropout(mfcc_network, p=self.net_opts['dropout_p']), 
                                        self.net_opts['num_class'], 
                                        nonlinearity=lasagne.nonlinearities.softmax)
        
        self.network = mfcc_network
        return self.network

class SpecCNNModel(DNNModel, SpecFeature):
    def build_network(self, mfcc_input_var):
        print('Building cnn with parameters:')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.net_opts)

        mfcc_network = layers.InputLayer((None, 130, MC_LENGTH), mfcc_input_var)
        mfcc_network = layers.BatchNormLayer(mfcc_network)
        mfcc_network = self.set_conv_layer(mfcc_network, 'conv_1', bnorm=False)
        mfcc_network = self.set_pool_layer(mfcc_network, 'pool_1')
        mfcc_network = self.set_conv_layer(mfcc_network, 'conv_2', bnorm=False)
        mfcc_network = self.set_pool_layer(mfcc_network, 'pool_2')
        for n in self.net_opts['layer_list']:
            # mfcc_network = layers.batch_norm(layers.DenseLayer(layers.dropout(mfcc_network, p=self.net_opts['dropout_p']), 
            #                                  n, 
            #                                  nonlinearity=lasagne.nonlinearities.rectify)
            #                                 )
            mfcc_network = layers.DenseLayer(layers.dropout(mfcc_network, p=self.net_opts['dropout_p']), 
                                            n, 
                                            nonlinearity=lasagne.nonlinearities.rectify)
            # mfcc_network = layers.BatchNormLayer(mfcc_network)
        mfcc_network = layers.DenseLayer(layers.dropout(mfcc_network, p=self.net_opts['dropout_p']), 
                                        self.net_opts['num_class'], 
                                        nonlinearity=lasagne.nonlinearities.softmax)
        
        self.network = mfcc_network
        return self.network

class CocktailCNNModel(DNNModel, CocktailFeature):
    def build_network(self, mfcc_input_var):
        print('Building cnn with parameters:')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.net_opts)

        mfcc_network = layers.InputLayer((None, 169, MC_LENGTH), mfcc_input_var)
        mfcc_network = layers.BatchNormLayer(mfcc_network)
        mfcc_network = self.set_conv_layer(mfcc_network, 'conv_1')
        mfcc_network = self.set_pool_layer(mfcc_network, 'pool_1')
        mfcc_network = self.set_conv_layer(mfcc_network, 'conv_2')
        mfcc_network = self.set_pool_layer(mfcc_network, 'pool_2')
        # mfcc_network = self.set_conv_layer(mfcc_network, 'conv_3')
        # mfcc_network = self.set_pool_layer(mfcc_network, 'pool_3')
        for n in self.net_opts['layer_list']:
            mfcc_network = layers.DenseLayer(layers.dropout(mfcc_network, p=self.net_opts['dropout_p']), 
                                            n, 
                                            nonlinearity=lasagne.nonlinearities.rectify)
        mfcc_network = layers.DenseLayer(layers.dropout(mfcc_network, p=self.net_opts['dropout_p']), 
                                        self.net_opts['num_class'], 
                                        nonlinearity=lasagne.nonlinearities.softmax)
        
        self.network = mfcc_network
        return self.network

    

##### Raw DNN 
class RawDNNModel(Model, RawFeature):
    def init_model(self):
        print('Initializing model...')
        ra_input_var = T.tensor3('raw_audio_input')
        mc_input_var = T.tensor3('melody_contour_input')
        target_var = T.imatrix('targets')
        network = self.build_network(ra_input_var, mc_input_var)
        prediction = layers.get_output(network)
        prediction = T.clip(prediction, 1e-7, 1.0 - 1e-7)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        params = layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.adagrad(loss, params, learning_rate=0.002)
        # updates = lasagne.updates.sgd(loss, params, learning_rate=0.02)

        test_prediction = layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)),
                          dtype=theano.config.floatX)

        print('Building functions...')
        self.train_fn = theano.function([ra_input_var, mc_input_var, target_var], 
                                        [loss, prediction], 
                                        updates=updates, 
                                        on_unused_input='ignore')
        self.val_fn = theano.function([ra_input_var, mc_input_var, target_var], 
                                        [test_loss, test_acc, test_prediction], 
                                        on_unused_input='ignore')
        self.run_fn = theano.function([ra_input_var, mc_input_var],
                                        [prediction],
                                        on_unused_input='ignore')

    def build_network(self, ra_input_var, mc_input_var):
        print('Building raw dnn with parameters:')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.net_opts)

        ra_network_1 = layers.InputLayer((None, 1, 3969), ra_input_var)
        ra_network_1 = self.set_conv_layer(ra_network_1, 'ra_conv_1', dropout=False, pad='same')
        ra_network_1 = self.set_pool_layer(ra_network_1, 'ra_pool_1')
        ra_network_1 = self.set_conv_layer(ra_network_1, 'ra_conv_2', pad='same')
        ra_network_1 = self.set_pool_layer(ra_network_1, 'ra_pool_2')
        ra_network_1 = self.set_conv_layer(ra_network_1, 'ra_conv_3', pad='same')
        ra_network_1 = self.set_pool_layer(ra_network_1, 'ra_pool_3')
        ra_network_1 = self.set_conv_layer(ra_network_1, 'ra_conv_4', pad='same')
        ra_network_1 = self.set_pool_layer(ra_network_1, 'ra_pool_4')
        concat_list = [ra_network_1]
        mc_input = layers.InputLayer((None, 2, MC_LENGTH), mc_input_var)
        concat_list.append(mc_input)
        network = layers.ConcatLayer(concat_list, axis=1, cropping=[None, None, 'center'])
        network = layers.BatchNormLayer(network)
        for n in self.net_opts['layer_list']:
            network = layers.DenseLayer(layers.dropout(network, p=self.net_opts['dropout_p']), 
                                            n, 
                                            nonlinearity=lasagne.nonlinearities.rectify)
        network = layers.DenseLayer(layers.dropout(network, p=self.net_opts['dropout_p']), 
                                        self.net_opts['num_class'], 
                                        nonlinearity=lasagne.nonlinearities.softmax)
        
        # print(layers.get_output_shape(network))
        self.network = network
        return self.network

    def test(self, feature_list):
        test_err = 0
        test_acc = 0
        test_batches = 0
        ans_list, pred_list = [], []
        for bt in self.iterate_minibatches(feature_list, 10):
            raw, mc, ans, fn = list(zip(*bt))
            err, acc, pred = self.val_fn(list(raw), list(mc), list(ans))
            test_err += err
            test_acc += acc
            test_batches += 1
            for a, p in zip(ans, pred):
                ans_list.append(np.argmax(a))
                pred_list.append(np.argmax(p))
        confusion_mat = confusion_matrix(ans_list, pred_list)
        print("  test loss:\t\t{:.6f}".format(old_div(test_err, test_batches)))
        print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
        print('confusion matrix:')
        print(confusion_mat)
        return confusion_mat

    def train_one(self, train_list):
        train_err = 0
        train_batches = 0
        for bt in self.iterate_minibatches(train_list, 10):
            raw, mc, ans, fn = list(zip(*bt))
            err, pred = self.train_fn(list(raw), list(mc), list(ans))
            # if random.randint(0, 19) == 0:
            #   print 'pred', pred
                # print 'll', ll
            train_err += err
            train_batches += 1
        return train_err, train_batches

    def val_one(self, val_list):
        val_err = 0
        val_acc = 0
        val_batches = 0
        for bt in self.iterate_minibatches(val_list, 10):
            raw, mc, ans, fn = list(zip(*bt))
            err, acc, pred = self.val_fn(list(raw), list(mc), list(ans))
            val_err += err
            val_acc += acc
            val_batches += 1
        return val_err, val_acc, val_batches 

##### Raw Network
class RawNetModel(Model, RawFeature):
    def init_model(self):
        print('Initializing model...')
        ra_input_var = T.tensor3('raw_audio_input')
        mc_input_var = T.tensor3('melody_contour_input')
        target_var = T.imatrix('targets')
        network = self.build_network(ra_input_var, mc_input_var)
        prediction = layers.get_output(network)
        prediction = T.clip(prediction, 1e-7, 1.0 - 1e-7)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        params = layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.sgd(loss, params, learning_rate=0.02)

        test_prediction = layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)),
                          dtype=theano.config.floatX)

        print('Building functions...')
        self.train_fn = theano.function([ra_input_var, mc_input_var, target_var], 
                                        [loss, prediction], 
                                        updates=updates, 
                                        on_unused_input='ignore')
        self.val_fn = theano.function([ra_input_var, mc_input_var, target_var], 
                                        [test_loss, test_acc, test_prediction], 
                                        on_unused_input='ignore')
        self.run_fn = theano.function([ra_input_var, mc_input_var],
                                        [prediction],
                                        on_unused_input='ignore')

    def build_network(self, ra_input_var, mc_input_var):
        print('Building raw network with parameters:')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.net_opts)

        ra_network_1 = layers.InputLayer((None, 1, None), ra_input_var)
        ra_network_1 = self.set_conv_layer(ra_network_1, 'ra_conv_1', dropout=False, pad='same')
        ra_network_1 = self.set_pool_layer(ra_network_1, 'ra_pool_1')
        ra_network_1 = self.set_conv_layer(ra_network_1, 'ra_conv_2', pad='same')
        ra_network_1 = self.set_pool_layer(ra_network_1, 'ra_pool_2')
        ra_network_1 = self.set_conv_layer(ra_network_1, 'ra_conv_3', pad='same')
        ra_network_1 = self.set_pool_layer(ra_network_1, 'ra_pool_3')
        ra_network_1 = self.set_conv_layer(ra_network_1, 'ra_conv_4', pad='same')
        ra_network_1 = self.set_pool_layer(ra_network_1, 'ra_pool_4')
        concat_list = [ra_network_1]
        mc_input = layers.InputLayer((None, 2, None), mc_input_var)
        concat_list.append(mc_input)
        network = layers.ConcatLayer(concat_list, axis=1, cropping=[None, None, 'center'])
        network = self.set_conv_layer(network, 'conv_1')
        network = self.set_pool_layer(network, 'pool_1')
        network = self.set_conv_layer(network, 'conv_2')
        network = self.set_pool_layer(network, 'pool_2')
        network = self.set_conv_layer(network, 'conv_3')
        network = layers.GlobalPoolLayer(network, getattr(T, self.net_opts['global_pool_func']))
        # print(layers.get_output_shape(network))
        # network = layers.DenseLayer(layers.dropout(network, p=self.net_opts['dropout_p']), 
        #                           self.net_opts['dens_1'], 
        #                           nonlinearity=lasagne.nonlinearities.rectify)
        network = layers.DenseLayer(layers.dropout(network, p=self.net_opts['dropout_p']), 
                                    self.net_opts['dens_2'], 
                                    nonlinearity=lasagne.nonlinearities.rectify)
        network = layers.DenseLayer(layers.dropout(network, p=self.net_opts['dropout_p']), 
                                    self.net_opts['num_class'], 
                                    nonlinearity=lasagne.nonlinearities.softmax)
        # print(layers.get_output_shape(network))
        self.network = network
        return self.network

    def test(self, feature_list):
        test_err = 0
        test_acc = 0
        test_batches = 0
        ans_list, pred_list = [], []
        for ra, mc, ans, fi in feature_list:
            ra = ra.reshape((1,1,-1))
            mc = mc.reshape((1,2,-1))
            ans = np.array([ans], dtype='int32')
            err, acc, pred = self.val_fn(ra, mc, ans)
            test_err += err
            test_acc += acc
            test_batches += 1
            for a, p in zip(ans, pred):
                ans_list.append(np.argmax(a))
                pred_list.append(np.argmax(p))
        confusion_mat = confusion_matrix(ans_list, pred_list)
        print("  test loss:\t\t{:.6f}".format(old_div(test_err, test_batches)))
        print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
        print('confusion matrix:')
        print(confusion_mat)
        return confusion_mat

    def train_one(self, train_list):
        train_err = 0
        train_batches = 0
        for ra, mc, ans, fi in train_list:
            ra = ra.reshape((1,1,-1))
            mc = mc.reshape((1,2,-1))
            ans = np.array([ans], dtype='int32')
            err, pred = self.train_fn(ra, mc, ans)
            # if random.randint(0, 19) == 0:
            #   print 'pred', pred
                # print 'll', ll
            train_err += err
            train_batches += 1
        return train_err, train_batches

    def val_one(self, val_list):
        val_err = 0
        val_acc = 0
        val_batches = 0
        for ra, mc, ans, fi in val_list:
            ra = ra.reshape((1,1,-1))
            mc = mc.reshape((1,2,-1))
            ans = np.array([ans], dtype='int32')
            err, acc, pred = self.val_fn(ra, mc, ans)
            val_err += err
            val_acc += acc
            val_batches += 1
        return val_err, val_acc, val_batches 

# class OldRawNetModel(Model):
#   def init_model(self):
#       print('Initializing model...')
#       ra_input_var = T.tensor3('raw_audio_input')
#       mc_input_var = T.tensor3('melody_contour_input')
#       target_var = T.imatrix('targets')
#       network = self.build_network(ra_input_var, mc_input_var)
#       prediction = layers.get_output(network)
#       prediction = T.clip(prediction, 1e-7, 1.0 - 1e-7)
#       loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
#       loss = loss.mean()
#       params = layers.get_all_params(network, trainable=True)
#       updates = lasagne.updates.sgd(loss, params, learning_rate=0.02)

#       test_prediction = layers.get_output(network, deterministic=True)
#       test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
#                                                               target_var)
#       test_loss = test_loss.mean()
#       test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), T.argmax(target_var, axis=1)),
#                         dtype=theano.config.floatX)

#       print('Building functions...')
#       self.train_fn = theano.function([ra_input_var, mc_input_var, target_var], 
#                                       [loss, prediction], 
#                                       updates=updates, 
#                                       on_unused_input='ignore')
#       self.val_fn = theano.function([ra_input_var, mc_input_var, target_var], 
#                                       [test_loss, test_acc, test_prediction], 
#                                       on_unused_input='ignore')

#   def build_network(self, ra_input_var, mc_input_var):
#       print('Building raw network with parameters:')
#       pp = pprint.PrettyPrinter(indent=4)
#       pp.pprint(self.net_opts)

#       ra_network_1 = layers.InputLayer((None, 1, None), ra_input_var)
#       ra_network_1 = self.set_conv_layer(ra_network_1, 'ra_conv_1', dropout=False)
#       # ra_network_1 = set_pool_layer(ra_network_1, 'ra_pool_1')
#       ra_network_1 = self.set_conv_layer(ra_network_1, 'ra_conv_2')
#       # ra_network_2 = layers.InputLayer((None, 1, None), ra_input_var)
#       # ra_network_2 = set_conv_layer(ra_network_2, 'ra_conv_1_2', pad='same')
#       # ra_network_2 = set_conv_layer(ra_network_2, 'ra_conv_2')
#       # ra_network_3 = layers.InputLayer((None, 1, None), ra_input_var)
#       # ra_network_3 = set_conv_layer(ra_network_3, 'ra_conv_1_3', pad='same')
#       # ra_network_3 = set_conv_layer(ra_network_3, 'ra_conv_2')
#       # concat_list = [ra_network_1, ra_network_2, ra_network_3]
#       concat_list = [ra_network_1]
#       mc_input = layers.InputLayer((None, 2, None), mc_input_var)
#       concat_list.append(mc_input)
#       network = layers.ConcatLayer(concat_list, axis=1, cropping=[None, None, 'center'])
#       network = self.set_conv_layer(network, 'conv_1')
#       network = self.set_pool_layer(network, 'pool_1')
#       network = self.set_conv_layer(network, 'conv_2')
#       network = self.set_pool_layer(network, 'pool_2')
#       network = self.set_conv_layer(network, 'conv_3')
#       network = layers.GlobalPoolLayer(network, self.net_opts['global_pool_func'])
#       # print(layers.get_output_shape(network))
#       network = layers.DenseLayer(layers.dropout(network, p=self.net_opts['dropout_p']), 
#                                   self.net_opts['num_class'], 
#                                   nonlinearity=lasagne.nonlinearities.softmax)
#       # print(layers.get_output_shape(network))
#       self.network = network
#       return self.network

#   def test(self, feature_list):
#       test_err = 0
#       test_acc = 0
#       test_batches = 0
#       ans_list, pred_list = [], []
#       for ra, mc, ans, fi in feature_list:
#           ra = ra.reshape((1,1,-1))
#           mc = mc.reshape((1,2,-1))
#           ans = np.array([ans], dtype='int32')
#           err, acc, pred = self.val_fn(ra, mc, ans)
#           test_err += err
#           test_acc += acc
#           test_batches += 1
#           for a, p in zip(ans, pred):
#               ans_list.append(np.argmax(a))
#               pred_list.append(np.argmax(p))
#       confusion_mat = confusion_matrix(ans_list, pred_list)
#       print("  test loss:\t\t{:.6f}".format(test_err / test_batches))
#       print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))
#       print('confusion matrix:')
#       print(confusion_mat)
#       return confusion_mat

#   def train_one(self, train_list):
#       train_err = 0
#       train_batches = 0
#       for ra, mc, ans, fi in train_list:
#           ra = ra.reshape((1,1,-1))
#           mc = mc.reshape((1,2,-1))
#           ans = np.array([ans], dtype='int32')
#           err, pred = self.train_fn(ra, mc, ans)
#           # if random.randint(0, 19) == 0:
#           #   print 'pred', pred
#               # print 'll', ll
#           train_err += err
#           train_batches += 1
#       return train_err, train_batches

#   def val_one(self, val_list):
#       val_err = 0
#       val_acc = 0
#       val_batches = 0
#       for ra, mc, ans, fi in val_list:
#           ra = ra.reshape((1,1,-1))
#           mc = mc.reshape((1,2,-1))
#           ans = np.array([ans], dtype='int32')
#           err, acc, pred = self.val_fn(ra, mc, ans)
#           val_err += err
#           val_acc += acc
#           val_batches += 1
#       return val_err, val_acc, val_batches 

