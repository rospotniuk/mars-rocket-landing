import os
import random
import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K


# Create a Keras Huber loss function
def hubert_loss(y_true, y_pred):
    err = y_pred - y_true
    return K.mean(K.sqrt(1+K.square(err))-1, axis=-1)


class DQNAgent(object):
    def __init__(self,
                 action_size, state_size,
                 layers=[50, 40],
                 learning_rate=0.001, gamma=0.95,
                 exploration_rate=1.0, exploration_rate_min=0.01, exploration_decay=0.995,
                 model_weights_path="mars_landing_weigths.h5"
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
        self.model = self._build_model()
        self.memory = deque(maxlen=2500)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.layers[0], input_dim=self.state_size, activation="relu"))
        for hidden_layer_size in self.layers[1:]:
            model.add(Dense(hidden_layer_size, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss=hubert_loss, optimizer=Adam(lr=self.learning_rate))

        if os.path.isfile(self.model_weights_path):
            model.load_weights(self.model_weights_path)
            self.exploration_rate = self.exploration_rate_min
        return model

    def save_model(self):
        self.model.save(self.model_weights_path)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def act(self, state):
        if (np.random.uniform(0, 1) <= self.exploration_rate):
            return np.random.randint(0, self.action_size)
        action = self.model.predict(state)
        return np.argmax(action)

    def partial_fit(self, batch_size=32):
        # Do not learn if number of observations in buffer is low
        loss = 0.0
        if len(self.memory) < batch_size:
            return loss
        # Sample a list of `batch_size` random indices from the memory without replacement
        sample_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in sample_batch:
            # We can improve only the target for the action in the observation <S, A, R, S'>
            target_for_action = reward   # It is correct if the state is final
            if not done:
                # If the state is not final, we add to it the discounted future rewards per current policy
                target_for_action += self.gamma * np.amax(self.model.predict(next_state)[0])
            target = self.model.predict(state)
            target[0][action] = target_for_action
            # Do one learning step (epoch=1) with the given (X, y) = (state, target)
            history = self.model.fit(state, target, epochs=1, verbose=0)
            loss += history.history['loss'][-1]
        # Each epoch (a training step) we will reduce the exploration rate until it is less than some value
        if self.exploration_rate > self.exploration_rate_min:
            self.exploration_rate *= self.exploration_decay
        # Return the average loss of this training step
        return loss / batch_size