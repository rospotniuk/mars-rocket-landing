import logging
import math
import random
from gym import spaces
import gym
import numpy as np
import time
import pyglet

logger = logging.getLogger(__name__)

# GLOBALS

PLATFORM_WIDTH = 0.25
PLATFORM_HEIGHT = 0.06

FRAME_TIME = 0.0167  # 60 fps

GRAVITY_ACCEL = 0.12
BOOST_ACCEL = 0.18
ROTATION_ACCEL = 20

W1 = 1.0  # X error reward weight
W2 = 0.001  # Y error reward weight
W3 = 0.0  # Theta error reward weight
W4 = 0.0  # Ydot error reward weight


class RocketEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self):

        self.action_space = spaces.Discrete(6)

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        self.isBoosting = False

    def _seed(self, seed=None):
        random.seed(seed)
        return [seed]

    def _step(self, inp):

        action = inp[0]
        num_steps = inp[1]

        # action = 0: no moves
        # action = 1: left turn
        # action = 2: right turn
        # action = 3: thrust
        # action = 4: left turn + thrust
        # action = 5: right turn + thrust

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        x, y, a, x_dot, y_dot, a_dot = state

        # Apply gravity
        y_dot += GRAVITY_ACCEL * FRAME_TIME

        if action == 3 or action == 4 or action == 5:
            self.isBoosting = True
            y_dot -= BOOST_ACCEL * FRAME_TIME * math.cos(a * 3.14159265 / 180)
            x_dot -= BOOST_ACCEL * FRAME_TIME * -math.sin(a * 3.14159265 / 180)
        else:
            self.isBoosting = False

        if action == 1 or action == 4:  # CCW
            a_dot -= ROTATION_ACCEL * FRAME_TIME

        if action == 2 or action == 5:  # CW
            a_dot += ROTATION_ACCEL * FRAME_TIME

        x += x_dot * FRAME_TIME
        y += y_dot * FRAME_TIME
        a += a_dot * FRAME_TIME

        #  Stay in bounds
        # if x < 0:  # Left
        #     done = True
        # elif x > 1.0:  # Right
        #     done = True
        # elif y < 0:  # Top
        #     done = True
        # if y > 1.0 - PLATFORM_HEIGHT:
        #     done = True
        # else:
        #     done = False

        if num_steps > 500:
            done = True
        elif y > 1.0 - PLATFORM_HEIGHT:
            done = True
        else:
            done = False

        self.state = (x, y, a, x_dot, y_dot, a_dot)

        x_error = (x - 0.5)**8
        y_error = (y - 1.0 + PLATFORM_HEIGHT)**2
        a_error = (a - 0)**2
        ydot_error = (y_dot - 0)**2

        reward = -(W1*x_error + W2*y_error + W3*a_error + W4*ydot_error)

        return np.array(self.state), reward, done, {}

    def _reset(self):

        x = random.uniform(0, 1)
        # x = 0.1
        y = 0.2
        a = random.randrange(start=-30, stop=30, step=1)
        x_dot = 0.0
        y_dot = 0.0
        a_dot = 0.0

        self.state = (x, y, a, x_dot, y_dot, a_dot)

        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400
        rocket_height = 60
        rocket_width = 6

        platform_width = PLATFORM_WIDTH * screen_width
        platform_height = PLATFORM_HEIGHT * screen_height

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = -rocket_width/2.0, rocket_width/2.0, rocket_height/2.0, -rocket_height/2.0
            self.rocket = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            if self.isBoosting:
                self.rocket.set_color(1.0, 0.0, 0.0)
            else:
                self.rocket.set_color(0.5, 0.5, 0.5)
            self.rockettrans = rendering.Transform()
            self.rocket.add_attr(self.rockettrans)
            self.viewer.add_geom(self.rocket)

            l, r, t, b = -platform_width/2.0, platform_width/2.0, platform_height, 0
            platform = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            platform.set_color(0.5, 0.5, 0.5)
            self.platformtrans = rendering.Transform(translation=(screen_width*0.5, 0))
            platform.add_attr(self.platformtrans)
            self.viewer.add_geom(platform)

        if self.state is None:
            return None

        if self.isBoosting:
            self.rocket.set_color(1.0, 0.0, 0.0)
        else:
            self.rocket.set_color(0.5, 0.5, 0.5)

        states = self.state
        rocketx = (states[0]*screen_width) + rocket_width / 2.0  # MIDDLE OF CART
        rockety = (1.0 - states[1])*screen_height + rocket_height / 2.0  # MIDDLE OF CART
        self.rockettrans.set_translation(rocketx, rockety)
        self.rockettrans.set_rotation(-states[2] * math.pi / 180)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render_fullcycle(self, state_history, action_history, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400
        rocket_height = 60
        rocket_width = 6

        platform_width = PLATFORM_WIDTH * screen_width
        platform_height = PLATFORM_HEIGHT * screen_height

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = -rocket_width/2.0, rocket_width/2.0, rocket_height/2.0, -rocket_height/2.0
            self.rocket = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            if self.isBoosting:
                self.rocket.set_color(1.0, 0.0, 0.0)
            else:
                self.rocket.set_color(0.5, 0.5, 0.5)
            self.rockettrans = rendering.Transform()
            self.rocket.add_attr(self.rockettrans)
            self.viewer.add_geom(self.rocket)

            l, r, t, b = -platform_width/2.0, platform_width/2.0, platform_height/2.0, -platform_height/2.0
            platform = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            platform.set_color(0.5, 0.5, 0.5)
            self.platformtrans = rendering.Transform(translation=(screen_width*0.5, platform_height*0.5))
            platform.add_attr(self.platformtrans)
            self.viewer.add_geom(platform)

        y_dot_last = 0
        frame = FRAME_TIME

        for state, action in zip(state_history, action_history):

            if frame < FRAME_TIME:
                time.sleep(FRAME_TIME-frame)

            start = time.time()

            x, y, a, x_dot, y_dot, a_dot = state

            if action == 3 or action == 4 or action == 5:
                self.rocket.set_color(1.0, 0.0, 0.0)
            else:
                self.rocket.set_color(0.5, 0.5, 0.5)

            rocketx = (x*screen_width) + rocket_width / 2.0  # MIDDLE OF CART
            rockety = (1.0 - y)*screen_height + rocket_height / 2.0  # MIDDLE OF CART
            self.rockettrans.set_translation(rocketx, rockety)
            self.rockettrans.set_rotation(-a * math.pi / 180)

            y_dot_last = y_dot

            frame = time.time() - start

            self.viewer.render(return_rgb_array=mode == 'rgb_array')


        return self.viewer.render(return_rgb_array=mode == 'rgb_array')