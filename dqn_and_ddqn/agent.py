#imports

import tensorflow as tf
import os
from gym import wrappers
import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) #error only
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import math
import glob
import io
import base64
from IPython.display import HTML
import random
from IPython import display as ipythondisplay
from IPython.display import clear_output






## Q network 
# Base network for DQN and DDQN

class Qnetwork(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(Qnetwork, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for units in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                units, activation='relu', kernel_initializer=tf.keras.initializers.he_normal()))
        self.output_layer = tf.keras.layers.Dense(num_actions)

    @tf.function
    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output


#Experience Buffer class for storing experiences

class ExperienceBuffer:
    def __init__(self, max_experiences, min_experiences):
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.experiences= []

    def add_experiences(self, exp):
        self.experiences.append(exp)
        if len(self.experiences) > self.max_experiences:
            self.experiences.pop(0)


## DQN agent

class DQNAgent:
    def __init__(self,train_net,gamma,batch_size,num_actions):
        self.train_net = train_net
        self.gamma = gamma
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.num_actions = num_actions


    def get_action(self,state,eps):
        if random.random() < eps:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.train_net(state.reshape(1, -1)))

    def train(self,exp_buffer):
    
        if len(exp_buffer.experiences) < exp_buffer.min_experiences:
            return 0

        ids = np.random.randint(low=0, high=len(exp_buffer.experiences), size=self.batch_size)
        states = np.array([exp_buffer.experiences[id_][0] for id_ in ids])
        actions = np.array([exp_buffer.experiences[id_][1] for id_ in ids])
        rewards = np.array([exp_buffer.experiences[id_][2] for id_ in ids])
        next_states = np.array([(np.zeros(4)if exp_buffer.experiences[id_][3] is None else exp_buffer.experiences[id_][3]) for id_ in ids])

        # Obtaining Q from states
        Q = self.train_net(states)
        # Obtaining Q prime from next states 
        Q_prime = self.train_net(next_states)

        # computation of target q 
        Q_t = Q.numpy()
        updates = rewards

        #valid ids those which doesn't have next states 0
        valid_idxs = np.array(next_states).sum(axis=1) != 0
        batch_idxs = np.arange(self.batch_size)

        updates[valid_idxs] += self.gamma * np.amax(Q_prime.numpy()[valid_idxs, :], axis=1)

        Q_t[batch_idxs, actions] = updates

        with tf.GradientTape() as tape:
            output = self.train_net(states)
            loss = self.mse(Q_t, output)
            gradients = tape.gradient(loss, self.train_net.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.train_net.trainable_variables))
        return loss

class DDQNAgent:
    def __init__(self,train_net,target_net,gamma,batch_size,num_actions,tau):
        self.train_net = train_net
        self.target_net = target_net
        self.gamma = gamma
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam()
        self.mse = tf.keras.losses.MeanSquaredError()
        self.num_actions = num_actions
        self.tau = tau


    def get_action(self,state,eps):
        if random.random() < eps:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.train_net(state.reshape(1, -1)))

    def train(self,exp_buffer):
    
        #Storing some amount of experiences before training
        if len(exp_buffer.experiences) < exp_buffer.min_experiences:
            return 0

        # random selection of ids from the experience replay buffer
        ids = np.random.randint(low=0, high=len(exp_buffer.experiences), size=self.batch_size)
        states = np.array([exp_buffer.experiences[id_][0] for id_ in ids])
        actions = np.array([exp_buffer.experiences[id_][1] for id_ in ids])
        rewards = np.array([exp_buffer.experiences[id_][2] for id_ in ids])
        next_states = np.array([(np.zeros(4)if exp_buffer.experiences[id_][3] is None else exp_buffer.experiences[id_][3]) for id_ in ids])

        #break

        # Obtaining Q values for current state
        Q_train= self.train_net(states)

        # Obtaining Q values for next state
        Q_train_prime = self.train_net(next_states)

        # target q , will be update later
        Q_train_t = Q_train.numpy()

        updates = rewards

        #ids those next states are non zero, those ids are only valid one
        valid_idxs = np.array(next_states).sum(axis=1) != 0
        batch_idxs = np.arange(self.batch_size)

        # Q value Update equation

        A_prime = np.argmax(Q_train_prime.numpy(), axis=1)
        Q_target = self.target_net(next_states)
        updates[valid_idxs] += self.gamma * Q_target.numpy()[batch_idxs[valid_idxs], A_prime[valid_idxs]]
        Q_train_t[batch_idxs, actions] = updates

        with tf.GradientTape() as tape:
            output = self.train_net(states)
            loss = self.mse(Q_train_t, output)
            gradients = tape.gradient(loss, self.train_net.trainable_variables)
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.train_net.trainable_variables))

        #copying train network into target network partially 
        for t, e in zip(self.target_net.trainable_variables, self.train_net.trainable_variables):
            t.assign(t * (1 - self.tau) + e * self.tau)

        return loss

class Gameagent:
    def __init__(self,environment,max_epsilon=1,min_epsilon = 0.01,lambda_ = 0.0005,gamma = 0.95, batch_size = 32, tau=0.08, max_experiences=400000,min_experiences = 96,hidden_units =[30,30], lr =0.001,num_episodes = 300):
        #Define global varaiables
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.lambda_ = lambda_
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

        self.hidden_units = hidden_units
        self.lr = lr
        self.env = environment
        self.env_name = self.env.unwrapped.spec.id

        self.num_actions = self.env.action_space.n

        self.num_states = len(self.env.observation_space.sample())

        self.num_episodes = num_episodes

        self.total_rewards = np.empty( self.num_episodes)
        self.total_loss = np.empty( self.num_episodes)
        
        #Experience buffer object
        self.exp_buffer = ExperienceBuffer(self.max_experiences, self.min_experiences)
        
        ## DQN object
        self.dqn = Qnetwork(self.num_states,  self.hidden_units,  self.num_actions)
        self.agent = DQNAgent( self.dqn, self.gamma, self.batch_size,  self.num_actions)
        
        ## Double DQN objects
        self.train_net = Qnetwork(self.num_states, self.hidden_units, self.num_actions)
        self.target_net = Qnetwork(self.num_states, self.hidden_units, self.num_actions)
        self.DDQagent = DDQNAgent(self.train_net,self.target_net,self.gamma,self.batch_size,self.num_actions,self.tau)
        
    ## Game play function
    # Function to show the game play 

    def show_video(self):
        mp4list = glob.glob('video/*.mp4')
        if len(mp4list) > 0:
            mp4 = mp4list[0]
            video = io.open(mp4, 'r+b').read()
            encoded = base64.b64encode(video)
            ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                        loop controls style="height: 400px;">
                        <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                     </video>'''.format(encoded.decode('ascii'))))
        else: 
            print("Could not find video")


    def wrap_env(self):
        self.env = Monitor(self.env, './video', force=True)
        return self.env

    ## Making video of testing phase of the environment

    def make_video(self,agent):
        self.env = self.wrap_env()
        rewards = 0
        steps = 0
        done = False
        state = self.env.reset()
        while not done:
            self.env.render()
            action = agent.get_action(state,0)
            state, reward, done, info= self.env.step(action)
            steps += 1
            rewards += reward
        print("Testing steps: {} rewards {}: ".format(steps, rewards))
    
    def visualise(self,train_steps,train_losses,train_rewards):
        fig = plt.figure(figsize=(16,5))
        loss_ax = fig.add_subplot(1,2,1)
        plt.plot(train_steps,train_losses)
        loss_ax.title.set_text('Loss plot')

        loss_ax.set_xlabel('Episodes')
        loss_ax.set_ylabel('Loss')

        rewards_ax = fig.add_subplot(1,2,2)
        plt.plot(train_steps,train_rewards)

        rewards_ax.title.set_text('Reward plot')

        rewards_ax.set_xlabel('Episode')
        rewards_ax.set_ylabel('Rewards')
        plt.show()

    def test(self,agent_name):
        if(agent_name == 'dqn'):
            self.make_video(self.agent)
        else:
            self.make_video(self.DDQagent)
        self.show_video()
        
    ## training loop for DQN

    def train_dqn(self):
        eps = self.max_epsilon
        train_steps = []
        train_losses = []
        train_rewards = []
        render = False
        steps = 0
        for i in range( self.num_episodes):
            clear_output(wait=True)
            rewards = 0
            state = self.env.reset()
            iteration = 0
            avg_loss = 0
            while True:
                action = self.agent.get_action(state,eps)
                next_state, reward, done, info = self.env.step(action)
                if done:
                    next_state = None
                # Experience Buffer replay
                self.exp_buffer.add_experiences((state, action, reward, next_state))

                loss = self.agent.train(self.exp_buffer)
                avg_loss += loss

                state = next_state

                # exponentially decay the eps value
                steps += 1
                eps = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-self.lambda_ * steps)

                if done:
                    avg_loss /= iteration
                    rewards = iteration
                    break
                iteration += 1
            train_steps.append(i)
            train_rewards.append(rewards)
            train_losses.append(float(avg_loss))
            print(f"Episode: {i}, Episode Reward: {rewards}, Episode Loss: {avg_loss: .3f}, Epsilon: {eps}")
        return train_steps,train_losses,train_rewards

    ## training loop for DDQN   
    def train_ddqn(self):
        eps = self.max_epsilon
        train_steps = []
        train_losses = []
        train_rewards = []
        render = False
        steps = 0
        for i in range( self.num_episodes):
            clear_output(wait=True)
            rewards = 0
            state = self.env.reset()
            iteration = 0
            avg_loss = 0
            while True:
                action = self.DDQagent.get_action(state,eps)
                next_state, reward, done, info = self.env.step(action)
                if done:
                    next_state = None
                # Experience Buffer replay
                self.exp_buffer.add_experiences((state, action, reward, next_state))

                loss = self.DDQagent.train(self.exp_buffer)
                avg_loss += loss

                state = next_state

                # exponentially decay the eps value
                steps += 1
                eps = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * math.exp(-self.lambda_ * steps)

                if done:
                    avg_loss /= iteration
                    rewards = iteration
                    break
                iteration += 1
            train_steps.append(i)
            train_rewards.append(rewards)
            train_losses.append(float(avg_loss))
            print(f"Episode: {i}, Episode Reward: {rewards}, Episode Loss: {avg_loss: .3f}, Epsilon: {eps}")
        return train_steps,train_losses,train_rewards
    
    ## saving training variables
    
    def save_training_variable(self,agent_name,train_steps,train_rewards,train_losses):
        path = "./assets/"+self.env_name+"/"+agent_name+"/training_variables/"
        try:
            os.makedirs(path)
        except OSError:
            print ("%s Directory already created" % path)
        else:
            print ("Successfully created the directory %s" % path)
        np.save(path+"train_steps",np.array(train_steps))
        np.save(path+"train_rewards",np.array(train_rewards))
        np.save(path+"train_losses",np.array(train_losses))
        
    ## loading training variables
    
    def load_training_variable(self,agent_name):
        
        train_steps = np.load("./assets/"+self.env_name+"/"+agent_name+"/training_variables/train_steps.npy")
        train_rewards = np.load("./assets/"+self.env_name+"/"+agent_name+"/training_variables/train_rewards.npy")
        train_losses = np.load("./assets/"+self.env_name+"/"+agent_name+"/training_variables/train_losses.npy")
        
        return train_steps,train_rewards,train_losses
    
    ## save weights
        
    def save_weights(self,agent_name):
        path = "./assets/"+self.env_name+"/"+agent_name+"/weights/"
        try:
            os.makedirs(path)
        except OSError:
            print ("%s Directory already created" % path)
        else:
            print ("Successfully created the directory %s" % path)
       
        if(agent_name == 'dqn'):
            self.dqn.save_weights(path+self.env_name+"_"+agent_name+".h5")
        else:
            self.train_net.save_weights(path+self.env_name+"_"+agent_name+".h5")
    
    ## load weights
    # Subclass implementation cannot be load until it called on some data atleast once. 
    # in order to load the weights, have to run model once.
    ## https://www.tensorflow.org/guide/keras/save_and_serialize
    
    def load_weights(self,agent_name):
        if(agent_name == 'dqn'):
            self.num_episodes = 1
            self.exp_buffer = ExperienceBuffer(1, 0)
            t, l, r, = self.train_dqn()
            clear_output(wait=True)
            self.dqn.load_weights("./assets/"+self.env_name+"/"+agent_name+"/weights/"+self.env_name+"_"+agent_name+".h5")
        else:
            self.num_episodes = 1
            self.exp_buffer = ExperienceBuffer(1, 0)
            t, l, r, = self.train_ddqn()
            clear_output(wait=True)
            self.train_net.load_weights("./assets/"+self.env_name+"/"+agent_name+"/weights/"+self.env_name+"_"+agent_name+".h5")
        
        
    