import numpy as np
import tensorflow as tf
import itertools

import dsr_pong.common.autoencoder as ae


class DSR:

    def __init__(self, gridSize = 28,
                 input_channels = 3,
                 number_of_actions = 4,
                 gamma = 0.99,
                 state_features_dim = 256,
                 encoder_filter_channels = [32, 64, 64],
                 encoder_filter_sizes = [3,3,3]):

        # MDP parameters
        actions = range(number_of_actions)

        self.autoenc = ae.Autoencoder(gridSize=gridSize,
                                 input_channels=input_channels,
                                 encoder_filter_channels=encoder_filter_channels,
                                 encoder_filter_sizes=encoder_filter_sizes,
                                      name= "Feature_Encoder",
                                      embedding_size= state_features_dim)

        self.state_features = self.autoenc.z
        self.X = self.autoenc.X
        self.reconstruction = self.autoenc.Y


        with tf.name_scope("Reward_network"):
            # Create the reward network
            self.W_rew = tf.Variable(tf.random_uniform(shape=(state_features_dim, 1), minval=-1.0 / np.sqrt(state_features_dim),
                                                  maxval=1.0 / np.sqrt(state_features_dim)),name= "Weights")
            self.b_rew = tf.Variable([0], dtype=tf.float32,name= "Bias")
            self.reward = tf.add(tf.matmul(self.state_features, self.W_rew), self.b_rew, name= "Reward")
            self.reward = tf.reshape(self.reward,[-1])
            print("Reward shape: %s" % self.reward.get_shape())

        # Create the successor feature networks
        succ_feature_representations_to_stack = []
        q_values_to_stack = []

        # Stop gradient from flowing back trough the sf net to the input
        sf_net_input = tf.stop_gradient(self.state_features, name="Stop_Gradient_For_SF")

        #keep track of all the weights
        self.weights_m_net = []

        #SF network
        with tf.name_scope("SF_network"):
            for a in actions:
                with tf.name_scope("SF_%s" % str(a)):

                    with tf.name_scope("FC_1"):
                        W_s_1 = tf.Variable(
                            tf.random_uniform(shape=(state_features_dim, 512), minval=-1.0 / np.sqrt(state_features_dim),
                                              maxval=1.0 / np.sqrt(state_features_dim)),name= "Weights")
                        b_s_1 = tf.Variable(np.zeros(512), dtype=tf.float32,name="Bias")
                        hidden_layer_1 = tf.nn.relu(tf.add(tf.matmul(sf_net_input, W_s_1), b_s_1))

                    with tf.name_scope("FC_2"):
                        W_s_2 = tf.Variable(
                            tf.random_uniform(shape=(512, 256), minval=-1.0 / np.sqrt(512), maxval=1.0 / np.sqrt(512)),name= "Weights")
                        b_s_2 = tf.Variable(np.zeros(256), dtype=tf.float32,name= "Bias")
                        hidden_layer_2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer_1, W_s_2), b_s_2))

                    with tf.name_scope("FC_3"):
                        W_s_3 = tf.Variable(tf.random_uniform(shape=(256, state_features_dim), minval=-1.0 / np.sqrt(256),
                                                              maxval=1.0 / np.sqrt(256)),name= "Weights")
                        b_s_3 = tf.Variable(np.zeros(state_features_dim), dtype=tf.float32,name= "Bias")
                        #Removed RELU to allow negative SR
                        output_layer = tf.add(tf.matmul(hidden_layer_2, W_s_3), b_s_3)

                    # Store the weights
                    self.weights_m_net.append(W_s_1)
                    self.weights_m_net.append(b_s_1)
                    self.weights_m_net.append(W_s_2)
                    self.weights_m_net.append(b_s_2)
                    self.weights_m_net.append(W_s_3)
                    self.weights_m_net.append(b_s_3)

                    # Add the output layer to the dict
                    succ_feature_representations_to_stack.append(output_layer)

                # Compute the qvalue for that succ feature
                with tf.name_scope("Q_%s" % str(a)):
                    q_value = tf.add(tf.matmul(output_layer, self.W_rew), self.b_rew)
                    q_value = tf.reshape(q_value,[-1])
                    q_values_to_stack.append(q_value)

            #stack such that shape is (batch,numA)
            self.q_values = tf.stack(q_values_to_stack, axis= 1)
            print("Q-Values shape: %s" % str(self.q_values.get_shape()))

            #stack such that shape is (batch, numA, featureDim)
            self.succ_feature_representations = tf.stack(succ_feature_representations_to_stack,axis=1)

        with tf.name_scope("Action_selection"):
            # Do the actual action selection
            self.action = tf.argmax(self.q_values, axis=1)
            print("Action stack shape: %s" % str(self.action.get_shape()))

        with tf.name_scope("Loss"):
            # Create the loss functions
            self.loss_a = self.autoenc.cost
            self.observed_reward = tf.placeholder(dtype=tf.float32, shape= [None], name="observed_reward")
            self.loss_r = tf.reduce_mean(self.huber_loss(self.observed_reward - self.reward),name="loss_r")

            self.succ_feature_target = tf.placeholder(dtype=tf.float32, shape=(None, state_features_dim),
                                                 name="observed_next_features")
            # placeholder for observed actions
            self.observed_action = tf.placeholder(dtype=tf.int32, shape= [None])
            #use one hot encoding
            observed_action_one_hot = tf.one_hot(self.observed_action,number_of_actions,on_value= 1.0,off_value=0.0)
            #increase the dimension
            observed_action_one_hot = tf.expand_dims(observed_action_one_hot,-1)
            print("Action One Hot: %s" % observed_action_one_hot.get_shape())

            self.succ_feature_for_action = tf.reduce_sum(self.succ_feature_representations * observed_action_one_hot, reduction_indices= 1)
            print("Reduced over actions: %s" % str(self.succ_feature_for_action.get_shape()))

            #Successor Feature loss
            self.loss_m = tf.reduce_mean(self.huber_loss(sf_net_input + gamma * self.succ_feature_target - self.succ_feature_for_action),name="loss_m")

            self.total_loss = tf.add_n([self.loss_a,self.loss_r,self.loss_m],name= "total_loss")

    def copy_weights(self, other_dsr):
        ops = []
        with tf.name_scope("Copy_Weights"):
            # Assign weigths from the reward network
            ops.append(self.W_rew.assign(other_dsr.W_rew))
            ops.append(self.b_rew.assign(other_dsr.b_rew))

            # Assign the weights from the m net
            for w, w_other in zip(self.weights_m_net, other_dsr.weights_m_net):
                ops.append(w.assign(w_other))

            # Assign weights from the autoencoder
            ops.extend(self.autoenc.copy_weights(other_dsr.autoenc))

        return ops

    def list_weights(self):
        w = []
        w.append(self.W_rew)
        w.append(self.b_rew)
        w.extend(self.weights_m_net)
        w.extend(self.autoenc.list_weights())
        return w

    def huber_loss(self,x, delta = 1.0):
        loss = tf.where(condition= tf.abs(x) < delta,
                        x = tf.square(x) * 0.5,
                        y = delta * (tf.abs(x) - 0.5 * delta))
        return loss

class DoubleDSR:

    def __init__(self, number_of_actions, feature_size):
        self.current_net = DSR(gridSize=84,number_of_actions= number_of_actions, input_channels= 4,state_features_dim= feature_size)
        self.prev_net = DSR(gridSize= 84, number_of_actions= number_of_actions, input_channels= 4, state_features_dim= feature_size)

    def switch(self):
        ops = self.prev_net.copy_weights(self.current_net)
        return tf.group(*ops)

class ReplicaDSR:

    def __init__(self,num_actions, feature_size, num_gpus, optimizer_model, optimizer_sr):

        self.prev_net = DSR(gridSize=84, number_of_actions=num_actions, input_channels=4,
                            state_features_dim=feature_size)
        self._current_net_towers = []
        tower_grad_model = []
        tower_grad_sr = []

        #init the replicas
        for i in range(num_gpus):
            worker = '/gpu:%d' % i
            #use the cpu as parameter server
            device_setter = tf.train.replica_device_setter(worker_device= worker,ps_device="/cpu:0",ps_tasks=1)
            with tf.variable_scope("dsr",reuse= (i!= 0)):
                with tf.name_scope("tower_%d" % i) as name_scope:
                    with tf.device(device_setter):
                        tower = DSR(gridSize= 84,number_of_actions= num_actions, input_channels= 4,state_features_dim= feature_size)

                        if i == 0:
                            self.current_net = tower

                        self._current_net_towers.append(tower)
                        grads_model = optimizer_model.compute_gradients(tower.loss_r + tower.loss_a,tower.list_weights())
                        grads_sr = optimizer_sr.compute_gradients(tower.loss_m,tower.list_weights())
                        #clip the model gradients
                        with tf.name_scope("clip_gradients"):
                            for i, (grad, var) in enumerate(grads_model):
                                if grad is not None:
                                    grads_model[i] = (tf.clip_by_norm(grad, 1), var)

                        tower_grad_model.append(grads_model)
                        tower_grad_sr.append(grads_sr)


        #init the training node
        with tf.device("/cpu:0"):
            with tf.name_scope("averaging_gradients"):
                grads_model = average_gradients(tower_grad_model)
                grads_sr = average_gradients(tower_grad_sr)

                self.train_model = optimizer_model.apply_gradients(grads_model)
                self.train_sr = optimizer_sr.apply_gradients(grads_sr)


    def generate_feed_dict(self,batch_s,batch_r,batch_a = None,targets = None):
        feed_dict = {}
        num_tower = len(self._current_net_towers)
        #split the state batch
        for tower , batch in zip(self._current_net_towers,np.array_split(batch_s,num_tower)):
            feed_dict[tower.X] = batch

        if not batch_a is None:
            #split the action batch
            for tower , batch in zip(self._current_net_towers,np.array_split(batch_a,num_tower)):
                feed_dict[tower.observed_action] = batch
        # split the reward batch
        for tower, batch in zip(self._current_net_towers, np.array_split(batch_r, num_tower)):
            feed_dict[tower.observed_reward] = batch
        if not targets is None:
            #split the target batch
            for tower, batch in zip(self._current_net_towers,np.array_split(targets,num_tower)):
                feed_dict[tower.succ_feature_target] = batch

        return feed_dict

    def switch(self):
        ops = self.prev_net.copy_weights(self.current_net)
        return tf.group(*ops)






if __name__ == "__main__":
    test_dsr = DoubleDSR(6)

    init = tf.global_variables_initializer()
    assign_op = test_dsr.switch()

    with tf.Session() as sess:
        sess.run(init)

        x_prev = test_dsr.prev_net.W_rew.eval(sess)
        x_cur = test_dsr.current_net.W_rew.eval(sess)

        print(np.alltrue(x_prev == x_cur))

        sess.run(fetches= assign_op)

        x_prev = test_dsr.prev_net.W_rew.eval(sess)
        x_cur  = test_dsr.current_net.W_rew.eval(sess)
        print(np.alltrue(x_prev == x_cur))



def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      if not g is None:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

    if len(grads) > 0:
        # Average over the 'tower' dimension.
        grad = tf.concat(values= grads,axis= 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
  return average_grads
