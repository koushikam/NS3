#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Mar 29 22:54:51 2018

@author: koushik
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 19:41:49 2018
@author: koushik
"""
import os
import numpy as np
import tensorflow as tf
import pylab as pl
from operator import itemgetter
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
from numpy import random
from math import cos, sin, pi,exp 

import random
''' import network module '''
from algorithms import Station_map,STA_clustering

#################################################################################################
# input data information for Convolutional layer
length = 30; 
width = 1; 
state_shape = [1,length]; #  we consider single array input

#########################################################################################
# function to get state , reward and etc

class Wifi_NW():
    """
    class UAV positoin is the function which calculates the reward based on the chosen position and  
    again to generate new UAV 
    states and also network parameters.
    """
    def __init__(self,STA_xloc,STA_yloc,AP_loc,area,nSTAs,nAPs,sim_Time,visualize,MCS,actions):
        self.STA_xloc =STA_xloc;
        self.STA_yloc =STA_yloc;
        self.AP_loc = AP_loc;
        self.area = area;
        self.nSTAs = nSTAs;
        self.nAPs = nAPs;
        self.sim_Time = sim_Time;
        self.visualize=visualize;
        self.MCS = MCS;
        self.actions=actions;
        

    def send_to_nw(self,action,DL,UL):
        os.chdir('/home/koushik/ns-3/ns-3.28/') # setting the NS3 directory path
        """
        Input:  1. action--> STA_map
                2. DL --> download rates of STAs
                3. UL --> upload rates of STAs
        Ouput: new state
        """
        STA_map = self.actions[action];
        filenum=5;# NS3 file number
        inputs = {
                  'AP_xloc' :  pd.Series(self.AP_loc[:,0]),
                  'AP_yloc' :  pd.Series(self.AP_loc[:,1]),
                  'STA_xloc':  pd.Series(self.STA_xloc),
                  'STA_yloc':  pd.Series(self.STA_yloc),
                  'STA_map' :  pd.Series(STA_map),
                  'UL_Rate' :  pd.Series(UL),
                  'DL_Rate' :  pd.Series(DL)   
                 }

        df1 = pd.DataFrame(inputs) 
        df1.to_csv('inputs.csv')
        
        ### Pass network parameters to NS3 and run network simulation
        if self.visualize:
            os.system('./waf --pyrun "scratch/NW_%s.py --nAPs=%s --nSTAs=%s --area=%s --simulationTime=%s --MCS=%s" --vis '
                  %(str(filenum),str(self.nAPs),str(self.nSTAs),str(self.area),str(self.sim_Time),str(self.MCS)))
        else:
            os.system('./waf --pyrun "scratch/NW_%s.py --nAPs=%s --nSTAs=%s --area=%s --simulationTime=%s --MCS=%s" '
                  %(str(filenum),str(self.nAPs),str(self.nSTAs),str(self.area),str(self.sim_Time),str(self.MCS)))
        return self._get_state(action),self._get_reward()

    def _get_reward(self):
        df = pd.read_csv('NW_th.csv') # read from the results stored in csv file
        Umean =  np.mean(df.values[:,-1]);
        Dmean =  np.mean(df.values[:,-2]);
        Usum = np.sum(df.values[:,-1]);
        Dsum = np.sum(df.values[:,-2]);
        return  (Dsum+Usum) # aggregated throughput: Uplink throughput + Downlink Throughput
    
    def _get_state(self,action):
        state = action;
        return state;
    
################################################################################################
''' code here for Experience Replay '''
''' 
    for every action in state 's' the node takes action 'a' and gets reward 'r' then moves to state 's'' . This is stored in a 
    memory called as experience replay as a tuple '<s,a,r,s'>'. When the node requires, the node takes Batch of 'B' previous 
    experiences and uses that to train the Deep neural network module to improve the learning performance 
'''
class ReplayMemory:
    def __init__(self, model_dir,num_actions,batch_size=128,memory_size=10000):
        self.model_dir = model_dir
        self.num_actions=num_actions        
        self.memory_size = memory_size
        self.actions = np.empty(shape = [self.memory_size], dtype = np.uint8)
        self.rewards = np.empty(shape = [self.memory_size], dtype = np.float64)
        self.prestate = np.empty(shape = [self.memory_size]+state_shape, dtype = np.float16)
        self.poststate = np.empty(shape = [self.memory_size]+state_shape, dtype = np.float16)
        self.q_values = np.zeros(shape=[self.memory_size, self.num_actions], dtype=np.float)
        self.batch_size = batch_size
        self.count = 0
        self.current = 0
        
    ''' add current state, next state, Q_values of next state, reward and action for to replay memory '''
    def add(self, prestate, Q_st,poststate, reward, action):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.prestate[self.current] = prestate
        self.poststate[self.current] = poststate
        self.q_values[self.current] = Q_st
        self.count = max(self.count, self.current + 1)        
        self.current = (self.current + 1) % self.memory_size        
   
    ''' take sample equal to batch size for updating learning model '''           
    def sample(self):
        B_range =  min(self.count,self.batch_size)
        indexes = np.random.choice(range(self.count),B_range,replace=False)
        prestate = self.prestate[indexes]
        poststate = self.poststate[indexes]
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        Qs = self.q_values[indexes]
        return prestate, Qs,poststate, actions, rewards
   
###################################################################################################

         
'''
    Class Linear Control Signal is used to select the optimal tuning parameters like learning rate etc according to the number  
    of iterations and current iteration by using Starting and ending values as the limiting values for such parameters 
'''
class LinearControlSignal:
    """
    A control signal that changes linearly over time.

    This is used to change e.g. the learning-rate for the optimizer
    of the Neural Network, as well as other parameters.
    
    TensorFlow has functionality for doing this, but it uses the
    global_step counter inside the TensorFlow graph, while we
    want the control signals to use a state-counter for the
    game-environment. So it is easier to make this in Python.
    """

    def __init__(self, start_value, end_value, num_iterations, repeat=False):
        """
        Create a new object.

        :param start_value:
            Start-value for the control signal.

        :param end_value:
            End-value for the control signal.

        :param num_iterations:
            Number of iterations it takes to reach the end_value
            from the start_value.

        :param repeat:
            Boolean whether to reset the control signal back to the start_value
            after the end_value has been reached.
        """

        # Store arguments in this object.
        self.start_value = start_value
        self.end_value = end_value
        self.num_iterations = num_iterations
        self.repeat = repeat

        # Calculate the linear coefficient.
        self._coefficient = (end_value - start_value) / num_iterations

    def get_value(self, iteration):
        """Get the value of the control signal for the given iteration."""

        if self.repeat:
            iteration %= self.num_iterations

        if iteration < self.num_iterations:
            value = iteration * self._coefficient + self.start_value
        else:
            value = self.end_value

        return value

##########################################################################################

class NeuralNetwork:
    """
    Creates a Neural Network for Reinforcement Learning (Q-Learning).
    Functions are provided for estimating Q-values from states of the
    game-environment, and for optimizing the Neural Network so it becomes
    better at estimating the Q-values.
    
    """

    def __init__(self, num_actions, replay_memory, use_pretty_tensor=False):
        """
        :param num_actions:
            Number of discrete actions for the game-environment.

        :param replay_memory: 
            Object-instance of the ReplayMemory-class.

        :param use_pretty_tensor:
            Boolean whether to use PrettyTensor (True) which must then be
            installed, or use the tf.layers API (False) which is already
            built into TensorFlow.
        """
        self.num_actions=num_actions
        # Whether to use the PrettyTensor API (True) or tf.layers (False).
        self.use_pretty_tensor = use_pretty_tensor

        # Replay-memory used for sampling random batches.
        self.replay_memory = replay_memory

        # Path for saving/restoring checkpoints.
        #self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")

        # Placeholder variable for inputting states into the Neural Network.
        # A state is a multi-dimensional array holding image-frames from
        # the game-environment.
        self.x = tf.placeholder(dtype=tf.float32, shape=[None] + state_shape, name='x')

        # Placeholder variable for inputting the learning-rate to the optimizer.
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[])

        # Placeholder variable for inputting the target Q-values
        # that we want the Neural Network to be able to estimate.
        self.q_values_new = tf.placeholder(tf.float32,
                                           shape=[None, self.num_actions],
                                           name='q_values_new')

        # This is a hack that allows us to save/load the counter for
        # the number of states processed in the game-environment.
        # We will keep it as a variable in the TensorFlow-graph
        # even though it will not actually be used by TensorFlow.
        self.count_states = tf.Variable(initial_value=0,
                                        trainable=False, dtype=tf.int64,
                                        name='count_states')

        # Similarly, this is the counter for the number of episodes.
        self.count_episodes = tf.Variable(initial_value=0,
                                          trainable=False, dtype=tf.int64,
                                          name='count_episodes')

        # TensorFlow operation for increasing count_states.
        self.count_states_increase = tf.assign(self.count_states,
                                               self.count_states + 1)

        # TensorFlow operation for increasing count_episodes.
        self.count_episodes_increase = tf.assign(self.count_episodes,
                                                 self.count_episodes + 1)

        # The Neural Network will be constructed in the following.
        # Note that the architecture of this Neural Network is very
        # different from that used in the original DeepMind papers,
        # which was something like this:
        # Input image:      84 x 84 x 4 (4 gray-scale images of 84 x 84 pixels).
        # Conv layer 1:     16 filters 8 x 8, stride 4, relu.
        # Conv layer 2:     32 filters 4 x 4, stride 2, relu.
        # Fully-conn. 1:    256 units, relu. (Sometimes 512 units).
        # Fully-conn. 2:    num-action units, linear.

        # The DeepMind architecture does a very aggressive downsampling of
        # the input images so they are about 10 x 10 pixels after the final
        # convolutional layer. I found that this resulted in significantly
        # distorted Q-values when using the training method further below.
        # The reason DeepMind could get it working was perhaps that they
        # used a very large replay memory (5x as big as here), and a single
        # optimization iteration was performed after each step of the game,
        # and some more tricks.

        # Initializer for the layers in the Neural Network.
        # If you change the architecture of the network, particularly
        # if you add or remove layers, then you may have to change
        # the stddev-parameter here. The initial weights must result
        # in the Neural Network outputting Q-values that are very close
        # to zero - but the network weights must not be too low either
        # because it will make it hard to train the network.
        # You can experiment with values between 1e-2 and 1e-3.
        init = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)

        if self.use_pretty_tensor:
            # This builds the Neural Network using the PrettyTensor API,
            # which is a very elegant builder API, but some people are
            # having problems installing and using it.

            import prettytensor as pt

            # Wrap the input to the Neural Network in a PrettyTensor object.
            x_pretty = pt.wrap(self.x)

            # Create the convolutional Neural Network using Pretty Tensor.
            with pt.defaults_scope(activation_fn=tf.nn.relu):
                self.q_values = x_pretty. \
                    conv2d(kernel=3, depth=20, stride=1, name='layer_conv1', weights=init). \
                    conv2d(kernel=2, depth=40, stride=1, name='layer_conv2', weights=init). \
                    flatten().\
                    fully_connected(size=180, name='layer_fc3', weights=init). \
                    fully_connected(size=self.num_actions, name='layer_fc_out', weights=init,
                                    activation_fn=None)
 

            # Loss-function which must be optimized. This is the mean-squared
            # error between the Q-values that are output by the Neural Network
            # and the target Q-values.
            self.loss = self.q_values.l2_regression(target=self.q_values_new)
        else:
            # This builds the Neural Network using the tf.layers API,
            # which is very verbose and inelegant, but should work for everyone.

            # Note that the checkpoints for Tutorial #16 which can be
            # downloaded from the internet only support PrettyTensor.
            # Although the Neural Networks appear to be identical when
            # built using the PrettyTensor and tf.layers APIs,
            # they actually create somewhat different TensorFlow graphs
            # where the variables have different names, which means the
            # checkpoints are incompatible for the two builder APIs.

            # Padding used for the convolutional layers.
            padding = 'SAME'

            # Activation function for all convolutional and fully-connected
            # layers, except the last.
            activation = tf.nn.relu

            # Reference to the lastly added layer of the Neural Network.
            # This makes it easy to add or remove layers.
            net = self.x

            # First convolutional layer.
            
            
            # Flatten output of the last convolutional layer so it can
            # be input to a fully-connected (aka. dense) layer.
            # TODO: For some bizarre reason, this function is not yet in tf.layers
            # TODO: net = tf.layers.flatten(net)
            #net = tf.contrib.layers.flatten(net)
            

            # Fourth fully-connected layer.
            net = tf.layers.dense(inputs=net, name='layer_fc4', units=180,
                                  kernel_initializer=init, activation=activation)

            # Final fully-connected layer.
            net = tf.layers.dense(inputs=net, name='layer_fc_out', units=num_actions,
                                  kernel_initializer=init, activation=None)

            # The output of the Neural Network is the estimated Q-values
            # for each possible action in the game-environment.
            self.q_values = net

            # TensorFlow has a built-in loss-function for doing regression:
            # self.loss = tf.nn.l2_loss(self.q_values - self.q_values_new)
            # But it uses tf.reduce_sum() rather than tf.reduce_mean()
            # which is used by PrettyTensor. This means the scale of the
            # gradient is different and hence the hyper-parameters
            # would have to be re-tuned. So instead we calculate the
            # L2-loss similarly to how it is done in PrettyTensor.
            squared_error = tf.square(self.q_values - self.q_values_new)
            sum_squared_error = tf.reduce_sum(squared_error, axis=1)
            self.loss = tf.reduce_mean(sum_squared_error)
            #self.loss = self.q_values.l2_regression(target=self.q_values_new)
        
        # Optimizer used for minimizing the loss-function.
        # Note the learning-rate is a placeholder variable so we can
        # lower the learning-rate as optimization progresses.
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # Used for saving and loading checkpoints.
        #self.saver = tf.train.Saver()

        # Create a new TensorFlow session so we can run the Neural Network.
        self.session = tf.Session()

        # Load the most recent checkpoint if it exists,
        # otherwise initialize all the variables in the TensorFlow graph.
        self.load_checkpoint()

    def close(self):
        """Close the TensorFlow session."""
        self.session.close()

    def load_checkpoint(self):
        """
        Load all variables of the TensorFlow graph from a checkpoint.
        If the checkpoint does not exist, then initialize all variables.
        """

        self.session.run(tf.global_variables_initializer())

    def save_checkpoint(self, current_iteration):
        """Save all variables of the TensorFlow graph to a checkpoint."""

        self.saver.save(self.session,
                        save_path=self.checkpoint_path,
                        global_step=current_iteration)

        print("Saved checkpoint.")

    def get_q_values(self, states):
        """
        Calculate and return the estimated Q-values for the given states.

        A single state contains two images (or channels): The most recent
        image-frame from the game-environment, and a motion-tracing image.
        See the MotionTracer-class for details.

        The input to this function is an array of such states which allows
        for batch-processing of the states. So the input is a 4-dim
        array with shape: [batch, height, width, state_channels].
        
        The output of this function is an array of Q-value-arrays.
        There is a Q-value for each possible action in the game-environment.
        So the output is a 2-dim array with shape: [batch, num_actions]
        """

        # Create a feed-dict for inputting the states to the Neural Network.
        feed_dict = {self.x: states}

        # Use TensorFlow to calculate the estimated Q-values for these states.
        values = self.session.run(self.q_values, feed_dict=feed_dict)

        return values

    def optimize(self, min_epochs=1.0, max_epochs=10,
                 batch_size=128, loss_limit=0.015,
                 learning_rate=1e-2,epsilon=0.3):
        """
        Optimize the Neural Network by sampling states and Q-values
        from the replay-memory.

        The original DeepMind paper performed one optimization iteration
        after processing each new state of the game-environment. This is
        an un-natural way of doing optimization of Neural Networks.

        So instead we perform a full optimization run every time the
        Replay Memory is full (or it is filled to the desired fraction).
        This also gives more efficient use of a GPU for the optimization.

        The problem is that this may over-fit the Neural Network to whatever
        is in the replay-memory. So we use several tricks to try and adapt
        the number of optimization iterations.

        :param min_epochs:
            Minimum number of optimization epochs. One epoch corresponds
            to the replay-memory being used once. However, as the batches
            are sampled randomly and biased somewhat, we may not use the
            whole replay-memory. This number is just a convenient measure.

        :param max_epochs:
            Maximum number of optimization epochs.

        :param batch_size:
            Size of each random batch sampled from the replay-memory.

        :param loss_limit:
            Optimization continues until the average loss-value of the
            last 100 batches is below this value (or max_epochs is reached).

        :param learning_rate:
            Learning-rate to use for the optimizer.
        """

#        print("Optimizing Neural Network to better estimate Q-values ...")
#        print("\tLearning-rate: {0:.1e}".format(learning_rate))
#        print("\tLoss-limit: {0:.3f}".format(loss_limit))
#        print("\tMax epochs: {0:.1f}".format(max_epochs))

        # Prepare the probability distribution for sampling the replay-memory.
        #self.replay_memory.prepare_sampling_prob(batch_size=batch_size)

        # Number of optimization iterations corresponding to one epoch.
        #iterations_per_epoch = self.replay_memory.num_used / batch_size
        iterations_per_epoch = 1;
        # Minimum number of iterations to perform.
        min_iterations = int(iterations_per_epoch * min_epochs)
        
        # Maximum number of iterations to perform.
        max_iterations = 50#int(iterations_per_epoch * max_epochs)

        # Buffer for storing the loss-values of the most recent batches.
        loss_history = np.zeros(self.num_actions, dtype=float)
       
        for i in range(max_iterations):
            # Randomly sample a batch of states and target Q-values
            # from the replay-memory. These are the Q-values that we
            # want the Neural Network to be able to estimate.
            
            #state_batch, q_values_batch = self.replay_memory.sample()
            state_batch, Q_tp1,STp1_batch, actions, rewards = self.replay_memory.sample()
            
            
            # target_q_t = self.discount * max_q_t_plus_1 +reward
            #feed_dict = {self.x=}
            q_values_batch = np.zeros(shape=[len(state_batch),self.num_actions],dtype=float)
            #for i in range(len(state_batch)):
                #Max_q_Tp1 = np.max(self.get_q_values(states=[STp1_batch[i]])[0])
            self.discount = epsilon;
            for i in range(len(state_batch)):
                q_values_batch[i,actions[i]] = rewards[i] + self.discount*np.max(Q_tp1[i])
            
            # Create a feed-dict for inputting the data to the TensorFlow graph.
            # Note that the learning-rate is also in this feed-dict.
            #loss_val = np.zeros(len(state_batch),dtype=float)
            #for i in range(len(state_batch)):
            feed_dict = {self.x: state_batch,
                         self.q_values_new: q_values_batch,
                         self.learning_rate: learning_rate}

                # Perform one optimization step and get the loss-value.
            loss_val, _ = self.session.run([self.loss, self.optimizer],
                                           feed_dict=feed_dict)

            # Shift the loss-history and assign the new value.
            # This causes the loss-history to only hold the most recent values.
            loss_history = np.roll(loss_history, 1)
            loss_history[0] = loss_val

            # Calculate the average loss for the previous batches.
            loss_mean = np.mean(loss_history)

            # Print status.
            #pct_epoch = i / iterations_per_epoch
            #msg = "\tBatch loss: {2:.4f}, Mean loss: {3:.4f}"
            #msg = msg.format(loss_val, loss_mean)
            

            # Stop the optimization if we have performed the required number
            # of iterations and the loss-value is sufficiently low.
            if i > min_iterations and loss_mean < loss_limit:
                break
            
        return loss_val, loss_mean

 
    def get_weights_variable(self, layer_name):
        """
        Return the variable inside the TensorFlow graph for the weights
        in the layer with the given name.

        Note that the actual values of the variables are not returned,
        you must use the function get_variable_value() for that.
        """

        if self.use_pretty_tensor:
            # PrettyTensor uses this name for the weights in a conv-layer.
            variable_name = 'weights'
        else:
            # The tf.layers API uses this name for the weights in a conv-layer.
            variable_name = 'kernel'

        with tf.variable_scope(layer_name, reuse=True):
            variable = tf.get_variable(variable_name)

        return variable

    def get_variable_value(self, variable):
        """Return the value of a variable inside the TensorFlow graph."""

        weights = self.session.run(variable)

        return weights

    def get_layer_tensor(self, layer_name):
        """
        Return the tensor for the output of a layer.
        Note that this does not return the actual values,
        but instead returns a reference to the tensor
        inside the TensorFlow graph. Use get_tensor_value()
        to get the actual contents of the tensor.
        """

        # The name of the last operation of a layer,
        # assuming it uses Relu as the activation-function.
        tensor_name = layer_name + "/Relu:0"

        # Get the tensor with this name.
        tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

        return tensor

    def get_tensor_value(self, tensor, state):
        """Get the value of a tensor in the Neural Network."""

        # Create a feed-dict for inputting the state to the Neural Network.
        feed_dict = {self.x: [state]}

        # Run the TensorFlow session to calculate the value of the tensor.
        output = self.session.run(tensor, feed_dict=feed_dict)

        return output

    def get_count_states(self):
        """
        Get the number of states that has been processed in the game-environment.
        This is not used by the TensorFlow graph. It is just a hack to save and
        reload the counter along with the checkpoint-file.
        """
        return self.session.run(self.count_states)

    def get_count_episodes(self):
        """
        Get the number of episodes that has been processed in the game-environment.
        """
        return self.session.run(self.count_episodes)

    def increase_count_states(self):
        """
        Increase the number of states that has been processed
        in the game-environment.
        """
        return self.session.run(self.count_states_increase)

    def increase_count_episodes(self):
        """
        Increase the number of episodes that has been processed
        in the game-environment.
        """
        return self.session.run(self.count_episodes_increase)

########################################################################

class EpsilonGreedy:
    """
    The epsilon-greedy policy either takes a random action with
    probability epsilon, or it takes the action for the highest
    Q-value.
    
    If epsilon is 1.0 then the actions are always random.
    If epsilon is 0.0 then the actions are always argmax for the Q-values.

    Epsilon is typically decreased linearly from 1.0 to 0.1
    and this is also implemented in this class.

    During testing, epsilon is usually chosen lower, e.g. 0.05 or 0.01
    """

    def __init__(self, num_actions,
                 epsilon_testing=0.05,
                 num_iterations=1e6,
                 start_value=1.0, end_value=0.1,
                 repeat=False):
        """
        
        :param num_actions:
            Number of possible actions in the game-environment.

        :param epsilon_testing:
            Epsilon-value when testing.

        :param num_iterations:
            Number of training iterations required to linearly
            decrease epsilon from start_value to end_value.
            
        :param start_value:
            Starting value for linearly decreasing epsilon.

        :param end_value:
            Ending value for linearly decreasing epsilon.

        :param repeat:
            Boolean whether to repeat and restart the linear decrease
            when the end_value is reached, or only do it once and then
            output the end_value forever after.
        """

        # Store parameters.
        self.num_actions = num_actions
        self.epsilon_testing = epsilon_testing

        # Create a control signal for linearly decreasing epsilon.
        self.epsilon_linear = LinearControlSignal(num_iterations=num_iterations,
                                                  start_value=start_value,
                                                  end_value=end_value,
                                                  repeat=repeat)

    def get_epsilon(self, iteration, training):
        """
        Return the epsilon for the given iteration.
        If training==True then epsilon is linearly decreased,
        otherwise epsilon is a fixed number.
        """

        if training:
            epsilon = self.epsilon_linear.get_value(iteration=iteration)
        else:
            epsilon = self.epsilon_testing

        return epsilon

    def get_action(self, q_values, iteration, training):
        """
        Use the epsilon-greedy policy to select an action.
        
        :param q_values:
            These are the Q-values that are estimated by the Neural Network
            for the current state of the game-environment.
         
        :param iteration:
            This is an iteration counter. Here we use the number of states
            that has been processed in the game-environment.

        :param training:
            Boolean whether we are training or testing the
            Reinforcement Learning agent.

        :return:
            action (integer), epsilon (float)
        """

        epsilon = self.get_epsilon(iteration=iteration, training=training)

        # With probability epsilon.
        if np.random.random() < epsilon:
            # Select a random action.
            action = np.random.randint(low=0, high=self.num_actions)
        else:
            # Otherwise select the action that has the highest Q-value.
            action = np.argmax(q_values)

        return action, epsilon

########################################################################


class Agent:
    
    def __init__(self,UL_R,DL_R,STA_xloc,STA_yloc,AP_loc,area,nSTAs,nAPs,sim_Time,visualize,MCS,n_actions,actions,epoch,epoch1,model_dir,render):
        
#         self.STA_xloc = STA_xloc;
#         self.STA_yloc = STA_yloc;
#         self.AP_loc = AP_loc;
#         self.area= area;
#         self.nSTAs = nSTAs;
#         self.nAPs = nAPs;
#         self.sim_Time=sim_Time;
#         self.visualize = visualize;
#         self.MCS = MCS;
        self.num_actions = n_actions;
        #self.actions = actions;
        self.DL = DL_R; # download bit rate
        self.UL = UL_R; # Upload bit rate
        
        ''' set up the UAV positioning parameters  '''
        self.env = Wifi_NW(STA_xloc,STA_yloc,AP_loc,area,nSTAs,nAPs,sim_Time,visualize,MCS,actions)
        
        #UAV_position(self.UAV_range,self.grid_size,self.Jamming)
        self.render = render;        
        self.epoch=epoch
        self.epoch1=epoch1
        self.training=True
        self.epsilon_greedy=EpsilonGreedy(start_value=1.0,
                                           end_value=0.01,
                                           num_iterations = self.epoch,
                                           num_actions=self.num_actions,
                                           epsilon_testing=0.01)
        
        self.learning_rate_control = LinearControlSignal(start_value=1e-3,
                                                         end_value=1e-5,
                                                         num_iterations = self.epoch)        
        ''' Replay memory initialization '''
        self.ReplayMemory = ReplayMemory(model_dir,self.num_actions,batch_size=256,memory_size=10000)   
        
        ''' Nural Network model initiazation '''
        self.model = NeuralNetwork(num_actions=self.num_actions,
                                   replay_memory=self.ReplayMemory)

    ''' Running UAV positioning model '''        
    def run(self,fig,ax):
        self.update_count=0;
        # initialize the reward 
        R1 = []#np.array([[0.0 for i in range(self.epoch)] for i in range(self.epoch1)])
        N_pos=30;        
        state_hist=[0]*length # to store UAV parameters in 3d tensor format
        ''' collect first 36 samples with random actions to form a complete state input '''
        input_t = np.resize(state_hist,(1,length)) # each state input is of shape (6,6,3)
        
             
        for e1 in range(self.epoch1):             
            for e in range(self.epoch-1):                 
                ''' input at t-1 time slot '''
                input_tm1 = input_t
                ''' determine the Q values for the current state input with learned training model 
                then using Epsilon greedy model determine the action '''
                q_values =  self.model.get_q_values(states=[input_tm1])                        
                action, epsilon = self.epsilon_greedy.get_action(q_values=q_values,
                                                                 iteration=e,
                                                                 training=self.training)
                ''' determine reward and new UAV position based on current action taken '''       
                pos_t,reward = self.env.send_to_nw(action,self.DL[e,:],self.UL[e,:])#(action=action)                
                R1.append(reward);                
                ''' update state history '''
                state_hist = np.append(state_hist,pos_t)
                ''' construct state input with updated UAV positions '''                
                input_t = np.resize(state_hist[-N_pos:],(1,length))
                ''' determine the Q value for input_t '''
                Q_tp1 = self.model.get_q_values(states=[input_t])[0]    
                
                ''' add datas to replay memory for future batch training of Nural network '''
                self.ReplayMemory.add(input_tm1,Q_tp1,input_t,reward,action)
                
                ''' updated learning rate for every iteration. Initially keep learning rate as 1 so that it learns greedily 
                then keep decreesing the value for every iteration '''
                learning_rate = self.learning_rate_control.get_value(iteration=e)
                    
                ''' Retrain the Neural Network learning model based on the past state, reward, actions '''
                loss_val, loss_mean = self.model.optimize(learning_rate,epsilon)
                if self.epoch%1==0:
                    print("\rEpoch {:03d}/{:03d} | Batch Loss {:.4f} | Mean Loss  {:.4f}".format(e+1,self.epoch,loss_val, loss_mean),end='\r')
                    msg = "Epsilon: {:.4f}| Reward: {:.4f} | action:{}"
                    #print(msg.format(epsilon,reward,action),end='\n')
                ''' plot the data at each eposch e1 '''
                ax.clear()
                ax.plot(R1)
                fig.canvas.draw()
                plt.grid();
                plt.legend(['Throughput_aggr'],loc=3)
                plt.ylabel('Mbps')
                plt.xlabel('Timeslot')
                plt.show()
                                
                    
        return R1

''' custom printing code 

x=['=']
for i in range(10):
    print('\r{}'.format(''.join((x*(i+1))+['>'])),end='\r')
    time.sleep(2)
    

'''
 
