import os
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K


# Create a Keras Huber loss function
def hubert_loss(y_true, y_pred):
    err = y_pred - y_true
    return K.mean(K.sqrt(1 + K.square(err)) - 1, axis=-1)


class Memory:
    def __init__(self):
        self.clear()

    @classmethod
    def push(cls, x, y):
        try:
            return np.concatenate((x, y), axis=0)
        except:
            return np.concatenate((x, np.array([y])), axis=0)

    @classmethod
    def shift(cls, x):
        try:
            return np.delete(x, 0, axis=0)
        except:
            return x

    def remember(self, S, A, R, S_, D):
        for attr, value in zip(("states", "actions", "rewards", "new_states", "dones"), (S, A, R, S_, D)):
            self.__dict__[attr] = Memory.push(self.__dict__[attr], value)

    def forget(self):
        for attr in ("states", "actions", "rewards", "new_states", "dones"):
            self.__dict__[attr] = Memory.shift(self.__dict__[attr])

    def clear(self):
        self.states = np.array([], dtype=np.float).reshape(0, 8)
        self.actions = np.array([], dtype=np.int)
        self.rewards = np.array([], dtype=np.float)
        self.new_states = np.array([], dtype=np.float).reshape(0, 8)
        self.dones = np.array([], dtype=np.int)



class DQNAgent(object):
    def __init__(self,
                 action_size=4, state_size=8,
                 layers=(512, 256),
                 learning_rate=0.001, gamma=0.975,
                 exploration_rate=1.0, exploration_rate_min=0.01, exploration_decay=0.995,
                 max_memory_len=50000,
                 model_weights_path="mars_landing_weigths.h5",
                 load_previous_weights=True
                 ):
        self.state_size = state_size
        self.action_size = action_size
        self.layers = layers
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.exploration_rate_min = exploration_rate_min
        self.exploration_decay = exploration_decay
        self.model_weights_path = model_weights_path
        self.load_previous_weights = load_previous_weights
        self.model = self._build_model()
        self.max_memory_len = max_memory_len
        self.memory = Memory()

    def _build_model(self, dropout=0.3, batch_norm=False):
        model = Sequential()
        model.add(Dense(self.layers[0], input_dim=self.state_size, activation="relu"))
        for hidden_layer_size in self.layers[1:]:
            model.add(Dense(hidden_layer_size, activation="relu"))
            model.add(Dropout(dropout))
            if batch_norm: model.add(BatchNormalization())
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss=hubert_loss, optimizer=Adam(lr=self.learning_rate))

        if self.load_previous_weights:
            if os.path.isfile(self.model_weights_path):
                model.load_weights(self.model_weights_path)
                self.exploration_rate = self.exploration_rate_min
            else:
                print("File '{}' does not exist. Retraining...".format(self.model_weights_path))
        return model

    def save_model(self):
        self.model.save(self.model_weights_path)

    def remember(self, state, action, reward, new_state, done):
        # If memory is full, then remove the first element
        self._forget()
        self.memory.remember(state, action, reward, new_state, done)

    def _forget(self):
        if np.alen(self.memory.states) >= self.max_memory_len:
            self.memory.forget()

    def act(self, state):
        if np.random.uniform(0, 1) <= self.exploration_rate:    # At the first episode we select each action randomly
            return np.random.randint(0, self.action_size)
        action = self.model.predict(state)
        return np.argmax(action)

    def partial_fit(self, batch_size=32, verbose=1):
        loss = 0.0
        n = self.memory.states.shape[0]
        # Do not learn if number of observations in buffer is low
        if n < batch_size:
            return loss
        # We can improve only the target for the action in the observation <S, A, R, S'>
        # If the state is not final, we add to it the discounted future rewards per current policy
        rewards = (self.memory.rewards - self.memory.rewards.mean()) / (self.memory.rewards.std() + np.finfo(np.float32).eps)
        targets_for_action = rewards + \
                    (1 - self.memory.dones) * self.gamma * np.amax(self.model.predict(self.memory.new_states), axis=1)
        targets = self.model.predict(self.memory.states)
        targets[np.arange(0, n), self.memory.actions] = targets_for_action
        # Do one learning step (epoch=1) with the given (X, y) = (states, targets)
        history = self.model.fit(self.memory.states, targets, epochs=1, verbose=verbose)
        loss += history.history['loss'][-1]
        # Each epoch (a training step) we will reduce the exploration rate until it is less than some value
        if self.exploration_rate > self.exploration_rate_min:
            self.exploration_rate *= self.exploration_decay
        # Return the loss of this training step
        return loss