from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import helper_tf_util
import time
import warnings





class Network:
    def __init__(self, dataset, config):
        flat_inputs = dataset.flat_inputs
        c = time.gmtime()
        self.config = config
        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path is None:
                self.saving_path = time.strftime('/'+str(self.config.csn)+'/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            else:
                self.saving_path = self.config.saving_path
            makedirs(self.saving_path) if not exists(self.saving_path) else None

        with tf.variable_scope('inputs'):
            self.inputs = dict()
            num_layers = self.config.num_layers
            self.inputs['xyz'] = flat_inputs[:num_layers]
            self.inputs['neigh_idx'] = flat_inputs[num_layers: 2 * num_layers]
            self.inputs['sub_idx'] = flat_inputs[2 * num_layers:3 * num_layers]
            self.inputs['interp_idx'] = flat_inputs[3 * num_layers:4 * num_layers]
            self.inputs['features'] = flat_inputs[4 * num_layers]
            self.inputs['labels'] = flat_inputs[4 * num_layers + 1]
            self.inputs['input_inds'] = flat_inputs[4 * num_layers + 2]
            self.inputs['cloud_inds'] = flat_inputs[4 * num_layers + 3]
            self.inputs['layer_labels'] = flat_inputs[4 * num_layers + 4: 5*num_layers + 4]

            self.labels = self.inputs['labels']
            self.is_training = tf.placeholder(tf.bool, shape=())
            self.training_step = 1
            self.training_epoch = 0
            self.correct_prediction = 0
            self.accuracy = 0
            self.mIou_list = [0]
            self.class_weights = DP.get_class_weights(dataset.name)
            self.Log_file = open('/'+str(self.config.csn)+'/log_train_' + dataset.name + str(dataset.val_split) + '.txt', 'a')

        with tf.variable_scope('layers'):
            self.logits, loss_knn = self.inference(self.inputs, self.is_training)

        #####################################################################
        # Ignore the invalid point (unlabeled) when calculating the loss #
        #####################################################################
        with tf.variable_scope('loss'):
            self.logits = tf.reshape(self.logits, [-1, config.num_classes])
            self.labels = tf.reshape(self.labels, [-1])

            # Boolean mask of points that should be ignored
            ignored_bool = tf.zeros_like(self.labels, dtype=tf.bool)
            for ign_label in self.config.ignored_label_inds:
                ignored_bool = tf.logical_or(ignored_bool, tf.equal(self.labels, ign_label))

            # Collect logits and labels that are not ignored
            valid_idx = tf.squeeze(tf.where(tf.logical_not(ignored_bool)))
            valid_logits = tf.gather(self.logits, valid_idx, axis=0)
            valid_labels_init = tf.gather(self.labels, valid_idx, axis=0)

            # Reduce label values in the range of logit shape
            reducing_list = tf.range(self.config.num_classes, dtype=tf.int32)
            inserted_value = tf.zeros((1,), dtype=tf.int32)
            for ign_label in self.config.ignored_label_inds:
                reducing_list = tf.concat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
            valid_labels = tf.gather(reducing_list, valid_labels_init)

            self.loss, self.loss_knn1= self.get_loss(valid_logits, valid_labels, self.class_weights, loss_knn)

        with tf.variable_scope('optimizer'):
            self.learning_rate = tf.Variable(config.learning_rate, trainable=False, name='learning_rate')
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            #self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_knn1)
            self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('results'):
            self.correct_prediction = tf.nn.in_top_k(valid_logits, valid_labels, 1)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            self.prob_logits = tf.nn.softmax(self.logits)

            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)

        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        c_proto = tf.ConfigProto()
        c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('/'+str(self.config.csn), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def inference(self, inputs, is_training):
        #print('inputs', inputs)
        d_out = self.config.d_out
        feature = inputs['features']
        feature = tf.layers.dense(feature, 8, activation=None, name='fc0')
        feature = tf.nn.leaky_relu(tf.layers.batch_normalization(feature, -1, 0.99, 1e-6, training=is_training))
        feature = tf.expand_dims(feature, axis=2)

        # ###########################Encoder############################
        
        f_encoder_list = []
        loss_knn = 0
        for i in range(self.config.num_layers):
            
            f_encoder_i, loss1 = self.dilated_res_block(feature, inputs['xyz'][i], inputs['neigh_idx'][i], d_out[i], inputs['layer_labels'][i],
                                                 'Encoder_layer_' + str(i), is_training)
            f_sampled_i = self.random_sample(f_encoder_i, inputs['sub_idx'][i])
            feature = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
            
        # ###########################Encoder############################

        feature = helper_tf_util.conv2d(f_encoder_list[-1], f_encoder_list[-1].get_shape()[3].value, [1, 1],
                                        'decoder_0',
                                        [1, 1], 'VALID', True, is_training)

        # ###########################Decoder############################
        f_decoder_list = []
        
        last_xyz = self.random_sample_xyz(inputs['xyz'][self.config.num_layers-1], inputs['sub_idx'][self.config.num_layers-1])

        for j in range(self.config.num_layers):
            #f_interp_i = self.nearest_interpolation(feature, inputs['interp_idx'][-j - 1])  
            f_interp_i = self.gather_3_neighbour(tf.squeeze(feature, axis=2), inputs['interp_idx'][-j - 1])
            feature_uped = f_encoder_list[-j - 2]
            feature_uped = tf.tile(feature_uped, [1, 1, 16, 1])

            xyz_high = last_xyz
            last_xyz = inputs['xyz'][-j-1]
            xyz_low = inputs['xyz'][-j-1]
            xyz_high = self.gather_3_neighbour(xyz_high, inputs['interp_idx'][-j-1])
            xyz_low = tf.expand_dims(xyz_low, 2)
            xyz_low = tf.tile(xyz_low, [1, 1, 16, 1])
            xyz_coder = self.rela_xyz_decoder(xyz_low, xyz_high)



            f_decoder_i, out_kind = helper_tf_util.conv2d_6210_transpose(tf.concat([feature_uped, f_interp_i], axis=3), xyz_coder, f_encoder_list[-j - 2].get_shape()[-1].value, [1, 1], 'Decoder_layer_' + str(j), [1, 1], 'VALID', bn=True, is_training=is_training)

            if j!=0:
                label_high = tf.expand_dims(inputs['layer_labels'][-j], -1)
                label_low = tf.expand_dims(inputs['layer_labels'][-j-1], -1)
                label_high = self.gather_3_neighbour(label_high, inputs['interp_idx'][-j - 1])
                #label_high = tf.expand_dims(label_high, 2)
                label_low = tf.expand_dims(label_low, 2)
                label_low = tf.tile(label_low, [1, 1, 16, 1])

                batch_size = tf.shape(label_high)[0]
                num_points = tf.shape(label_high)[1]
                num_neigh = tf.shape(label_high)[2]
                d = label_high.get_shape()[3].value

                lb_true = tf.constant(1, dtype = tf.int32, shape = [1, 1, 1, 1])
                lb_true = tf.tile(lb_true, [batch_size, num_points, num_neigh, d])
                lb_false = tf.constant(0, dtype = tf.int32, shape = [1, 1, 1, 1])
                lb_false = tf.tile(lb_false, [batch_size, num_points, num_neigh, d])
                labels_neigh = tf.where(tf.equal(label_low, label_high), lb_true, lb_false)

                # point kind loss
                f_knn_kind = tf.reshape(out_kind, [-1, 2])
                labels_neigh = tf.reshape(labels_neigh, [-1])
                
                labels_neigh = tf.one_hot(labels_neigh, depth=2)

                num_per_class = np.array([1.5, 3], dtype=np.float32)
                class_weights = tf.convert_to_tensor(num_per_class, dtype=tf.float32)     
                weights = tf.reduce_sum(class_weights * labels_neigh, axis=1)
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=f_knn_kind, labels=labels_neigh)
                losses = losses * weights
                loss_knn += tf.reduce_mean(losses)

            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        f_layer_fc1 = helper_tf_util.conv2d(f_decoder_list[-1], 64, [1, 1], 'fc1', [1, 1], 'VALID', True, is_training)
        f_layer_fc2 = helper_tf_util.conv2d(f_layer_fc1, 32, [1, 1], 'fc2', [1, 1], 'VALID', True, is_training)
        f_layer_drop = helper_tf_util.dropout(f_layer_fc2, keep_prob=0.5, is_training=is_training, scope='dp1')
        f_layer_fc3 = helper_tf_util.conv2d(f_layer_drop, self.config.num_classes, [1, 1], 'fc', [1, 1], 'VALID', False,
                                            is_training, activation_fn=None)
        f_out = tf.squeeze(f_layer_fc3, [2])
        return f_out, loss_knn


    def dilated_res_block(self, feature, xyz, neigh_idx, d_out, labels, name, is_training): 
        f_pc = helper_tf_util.conv2d(feature, d_out // 2, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        
        f_pc, loss1= self.building_weight_3d_block(xyz, f_pc, neigh_idx, d_out, labels, name + 'LFA', is_training)
       

        f_pc = helper_tf_util.conv2d(f_pc, d_out * 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training,
                                     activation_fn=None)
        shortcut = helper_tf_util.conv2d(feature, d_out * 2, [1, 1], name + 'shortcut', [1, 1], 'VALID',
                                         activation_fn=None, bn=True, is_training=is_training)
        return tf.nn.leaky_relu(f_pc + shortcut), loss1

   

    def building_weight_3d_block(self, xyz, feature, neigh_idx, d_out, labels, name, is_training):
        d_in = feature.get_shape()[-1].value
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)

        
        f_xyz = helper_tf_util.conv2d(f_xyz, d_in, [1, 1], name + 'mlp1', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(feature, axis=2), neigh_idx)
        f_pc_agg, f_knn_kind = self.att_weight_3d_pooling(feature, f_neighbours, f_xyz, d_out // 2, name + 'att_cosine_pooling_1', is_training)


        f_xyz = helper_tf_util.conv2d(f_xyz, d_out // 2, [1, 1], name + 'mlp2', [1, 1], 'VALID', True, is_training)
        f_neighbours = self.gather_neighbour(tf.squeeze(f_pc_agg, axis=2), neigh_idx)
        f_pc_agg, f_knn_kind = self.att_weight_3d_pooling(f_pc_agg, f_neighbours, f_xyz, d_out, name + 'att_cosine_pooling_2', is_training)
        

        kind_loss = 0
        return f_pc_agg, kind_loss
    
    @staticmethod
    def att_weight_3d_pooling(feature, f_feature, f_xyz, d_out, name, is_training):

        batch_size = tf.shape(f_xyz)[0]
        num_points = tf.shape(f_xyz)[1]
        num_neigh = tf.shape(f_xyz)[2]
        d = f_xyz.get_shape()[3].value

        # feature knn
        f_feature = tf.reshape(f_feature, shape=[-1, num_neigh, d])
        feature_knn = tf.layers.dense(f_feature, d, activation=None, use_bias=True, name=name + 'fc1')

        # feature center
        feature = tf.reshape(feature, shape=[-1, 1, d])
        feature = tf.layers.dense(feature, d , activation=None, use_bias=True, name=name + 'fc2')
        feature = tf.tile(feature, [1, num_neigh, 1])

        # xyz
        f_xyz = tf.reshape(f_xyz, shape=[-1, num_neigh, d])

        f_weight = tf.concat([feature_knn, f_xyz], axis=-1)
        f_new_feature = tf.concat([feature_knn, feature], axis=-1)

        # 6210
        # xyz weight
        f_xyz_weight = tf.layers.dense(f_xyz, 2*d, activation=tf.nn.leaky_relu, use_bias=True, name=name + 'fc_xyz')
        # feature weight
        f_feature_weight = tf.layers.dense(f_new_feature, 2*d, activation=tf.nn.leaky_relu, use_bias=True, name=name + 'fc')

        att_scores = f_xyz_weight + f_feature_weight
        #7501
        #att_scores = f_xyz_weight
        #7502
        #att_scores = f_feature_weight

        att_scores = tf.nn.softmax(att_scores, axis=1)
        f_agg = f_weight * att_scores
        f_agg = tf.reduce_sum(f_agg, axis=1)
        f_agg = tf.reshape(f_agg, [batch_size, num_points, 1, 2*d])

        f_agg = helper_tf_util.conv2d(f_agg, d_out, [1, 1], name + 'mlp', [1, 1], 'VALID', True, is_training)

        f_knn_kind = 0
        return f_agg, f_knn_kind

    
   

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
        xyz_tile = tf.tile(tf.expand_dims(xyz, axis=2), [1, 1, tf.shape(neigh_idx)[-1], 1])
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        relative_feature = tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
        return relative_feature
