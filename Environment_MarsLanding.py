import gym
from gym import spaces
from gym.envs.classic_control import rendering

import numpy as np
import random
import time


# GLOBALS

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400

ROCKET_WIDTH = 10
ROCKET_HEIGHT = 60

PLATO_SHIFT = 10

PLATFORM_WIDTH = 0.2 * SCREEN_WIDTH
PLATFORM_HEIGHT = 0.06  # !!!

TIME_FRAME = 0.0167  # 60 fps
LANDING_TIME = 2 # 2 sec

GRAVITY_ACCELERATION = 0.12 # 3.72
BOOST_ACCELERATION = 0.18 # 1.5 * GRAVITY_ACCELERATION 
ROTATION_ACCELERATION = 20

WIND_SPEED_RANGE = (0, 5)
WIND_SPEED_SHIFT = 1

RELIEF_COLOR = np.array([244, 164, 96]) / 255 # sandybrown
SKY_COLOR = np.array([135, 206, 235]) / 255 # skyblue

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
        self.plato_coords = {}
        self.relief_coords = self.__generate_relief()
        self.wind_speed = random.choice(WIND_SPEED_RANGE)
        self.wind_direction = random.choice((-1, 1)) # -1 means from right to left; 1 means from left to right


    def __generate_relief(self):
        pX = 0
        pY = 0
        coords = [(pX, pY), (pX, random.randint(int(0.667*SCREEN_HEIGHT), SCREEN_HEIGHT-25))]

        # The relief to the left to a plato
        left_points = random.choice((3, 5, 7))
        right_bound = int(0.5 * SCREEN_WIDTH)
        upper_bound = int(0.5 * SCREEN_HEIGHT)
        pX += 25
        for i, c in enumerate(range(left_points)):
            pX = random.randint(pX, right_bound)
            if c == left_points - 1:
                pY = random.randint(25, 125)
            else:
                pY = random.randint(25, upper_bound)
            coords.append((pX, pY))

        # Plato
        plato_width = random.randint(int(0.1*SCREEN_WIDTH), int(0.2*SCREEN_WIDTH))
        self.plato_coords["left"] = [pX + PLATO_SHIFT, pY]
        pX += plato_width
        self.plato_coords["right"] = [pX - PLATO_SHIFT, pY]
        coords.append((pX, pY))

        # The relief to the right to a plato
        right_points = random.choice((3, 5, 7))
        right_bound = int(SCREEN_WIDTH) - 25
        for i in range(right_points):
            pX = random.randint(pX, right_bound)
            pY = random.randint(25, upper_bound)
            coords.append((pX, pY))

        coords.extend([(SCREEN_WIDTH, random.randint(int(0.667*SCREEN_HEIGHT), SCREEN_HEIGHT-25)), (SCREEN_WIDTH, 0)])
        return coords


    def _seed(self, seed=None):
        random.seed(seed)
        np.random.random(seed)
        return [seed]


    def _step(self, inp):
        action = inp[0]
        num_steps = inp[1]

        # AVAILABLE ACTIONS:
        #   action = 0: no moves
        #   action = 1: left turn
        #   action = 2: right turn
        #   action = 3: thrust
        #   action = 4: left turn + thrust
        #   action = 5: right turn + thrust

        assert self.action_space.contains(action), "{} ({}) invalid".format(action, type(action))
        state = self.state
        x, z, theta, velX, velZ, velTheta = state

        # Take into account gravity
        velZ += GRAVITY_ACCELERATION * TIME_FRAME
        velX += self.wind_direction * (self.wind_speed + np.random.uniform(-WIND_SPEED_SHIFT, WIND_SPEED_SHIFT))

        if action == 3 or action == 4 or action == 5:
            self.isBoosting = True
            velZ -= BOOST_ACCELERATION * TIME_FRAME * np.cos(theta * np.pi / 180)
            velX -= -BOOST_ACCELERATION * TIME_FRAME * np.sin(theta * np.pi / 180)
        else:
            self.isBoosting = False

        if action == 1 or action == 4:  # CCW
            velTheta -= ROTATION_ACCELERATION * TIME_FRAME

        if action == 2 or action == 5:  # CW
            velTheta += ROTATION_ACCELERATION * TIME_FRAME

        x += velX * TIME_FRAME
        z += velZ * TIME_FRAME
        theta += velTheta * TIME_FRAME

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
        elif z > 1.0 - PLATFORM_HEIGHT:
            done = True
        else:
            done = False

        self.state = (x, z, theta, velX, velZ, velTheta)

        x_error = (x - 0.5)**8
        z_error = (z - 1.0 + PLATFORM_HEIGHT)**2
        theta_error = (theta - 0)**2
        velZ_error = (velZ - 0)**2

        reward = -(W1*x_error + W2*z_error + W3*theta_error + W4*velZ_error)

        return np.array(self.state), reward, done, {}


    def _reset(self):
        x = random.uniform(0, 1)
        z = 0.2
        theta = random.randrange(-30, 30)
        velX = 0.0
        velZ = 0.0
        velTheta = 0.0

        self.state = (x, z, theta, velX, velZ, velTheta)
        return np.array(self.state)


    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        platform_width = PLATFORM_WIDTH * screen_width
        platform_height = PLATFORM_HEIGHT * screen_height

        if self.viewer is None:
            self.viewer = rendering.Viewer(SCREEN_WIDTH, SCREEN_HEIGHT)
            # Sky
            sky = rendering.FilledPolygon([(0,0), (0,SCREEN_HEIGHT), (SCREEN_WIDTH,SCREEN_HEIGHT), (SCREEN_WIDTH,0)])
            sky.set_color(*SKY_COLOR)
            self.skytrans = rendering.Transform()
            sky.add_attr(self.skytrans)
            self.viewer.add_geom(sky)

            # Relief
            relief = rendering.FilledPolygon(self.relief_coords)
            relief.set_color(*RELIEF_COLOR)
            self.relieftrans = rendering.Transform()
            relief.add_attr(self.relieftrans)
            self.viewer.add_geom(relief)

            # Flags
            # Left one
            pillar_width = 2
            pillar_height = 25
            self.left_flag_pillar = rendering.FilledPolygon([
                (self.plato_coords["left"][0], self.plato_coords["left"][1]), 
                (self.plato_coords["left"][0], self.plato_coords["left"][1] + pillar_height), 
                (self.plato_coords["left"][0] + self.wind_direction * pillar_width, self.plato_coords["left"][1] + pillar_height), 
                (self.plato_coords["left"][0] + self.wind_direction * pillar_width, self.plato_coords["left"][1])
            ])
            self.left_flag_pillar.set_color(0, 0, 0)
            self.left_flag_pillar_trans = rendering.Transform()
            self.left_flag_pillar.add_attr(self.left_flag_pillar_trans)
            self.viewer.add_geom(self.left_flag_pillar)
            
            self.left_flag = rendering.FilledPolygon([
                (self.plato_coords["left"][0] + self.wind_direction * pillar_width, self.plato_coords["left"][1] + 10), 
                (self.plato_coords["left"][0] + self.wind_direction * pillar_width, self.plato_coords["left"][1] + pillar_height), 
                (self.plato_coords["left"][0] + self.wind_direction * (pillar_width + np.sqrt(3)/2 * 15), self.plato_coords["left"][1] + (pillar_height-10) / 2 + 10)
            ])
            if self.isBoosting:
                self.left_flag.set_color(1.0, 0.0, 0.0)
            else:
                self.left_flag.set_color(1.0, 1.0, 1.0)
            self.left_flag_trans = rendering.Transform()
            self.left_flag.add_attr(self.left_flag_trans)
            self.viewer.add_geom(self.left_flag)

            # Right one
            self.right_flag_pillar = rendering.FilledPolygon([
                (self.plato_coords["right"][0], self.plato_coords["right"][1]), 
                (self.plato_coords["right"][0], self.plato_coords["right"][1] + pillar_height), 
                (self.plato_coords["right"][0] + self.wind_direction * pillar_width, self.plato_coords["right"][1] + pillar_height), 
                (self.plato_coords["right"][0] + self.wind_direction * pillar_width, self.plato_coords["right"][1])
            ])
            self.right_flag_pillar.set_color(0, 0, 0)
            self.right_flag_pillar_trans = rendering.Transform()
            self.right_flag_pillar.add_attr(self.right_flag_pillar_trans)
            self.viewer.add_geom(self.right_flag_pillar)
            
            self.right_flag = rendering.FilledPolygon([
                (self.plato_coords["right"][0] + self.wind_direction * pillar_width, self.plato_coords["right"][1] + 10), 
                (self.plato_coords["right"][0] + self.wind_direction * pillar_width, self.plato_coords["right"][1] + pillar_height), 
                (self.plato_coords["right"][0] + self.wind_direction * (pillar_width + np.sqrt(3)/2 * 15), self.plato_coords["right"][1] + (pillar_height-10) / 2 + 10)
            ])
            if self.isBoosting:
                self.right_flag.set_color(1.0, 0.0, 0.0)
            else:
                self.right_flag.set_color(1.0, 1.0, 1.0)
            self.right_flag_trans = rendering.Transform()
            self.right_flag.add_attr(self.right_flag_trans)
            self.viewer.add_geom(self.right_flag)

            # Rocket
            l, r, t, b = -ROCKET_WIDTH/2.0, ROCKET_WIDTH/2.0, ROCKET_HEIGHT/2.0, -ROCKET_HEIGHT/2.0
            self.rocket = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.rockettrans = rendering.Transform()
            self.rocket.add_attr(self.rockettrans)
            self.viewer.add_geom(self.rocket)

        if self.state is None:
            return None
    """
        if self.isBoosting:
            self.left_flag.set_color(1.0, 0.0, 0.0)
            self.right_flag.set_color(1.0, 0.0, 0.0)
        else:
            self.right_flag.set_color(1.0, 1.0, 1.0)
            self.left_flag.set_color(1.0, 1.0, 1.0)
    """
        states = self.state
        rocketx = (states[0]*screen_width) + rocket_width / 2.0  # MIDDLE OF CART
        rockety = (1.0 - states[1])*screen_height + rocket_height / 2.0  # MIDDLE OF CART
        self.rockettrans.set_translation(rocketx, rockety)
        self.rockettrans.set_rotation(-states[2] * np.pi / 180)

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
        frame = TIME_FRAME

        for state, action in zip(state_history, action_history):

            if frame < TIME_FRAME:
                time.sleep(TIME_FRAME-frame)

            start = time.time()

            x, y, a, x_dot, y_dot, a_dot = state

            if action == 3 or action == 4 or action == 5:
                self.rocket.set_color(1.0, 0.0, 0.0)
            else:
                self.rocket.set_color(0.5, 0.5, 0.5)

            rocketx = (x*screen_width) + rocket_width / 2.0  # MIDDLE OF CART
            rockety = (1.0 - y)*screen_height + rocket_height / 2.0  # MIDDLE OF CART
            self.rockettrans.set_translation(rocketx, rockety)
            self.rockettrans.set_rotation(-a * np.pi / 180)

            y_dot_last = y_dot

            frame = time.time() - start

            self.viewer.render(return_rgb_array=mode == 'rgb_array')


        return self.viewer.render(return_rgb_array=mode == 'rgb_array')