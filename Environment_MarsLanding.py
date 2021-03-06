# This is an extended version of LunarLander-v2 (https://gym.openai.com/envs/LunarLander-v2/) openai.gym environment
# Main changes:
# 1.
#

import numpy as np
import random
import Box2D
from Box2D.b2 import edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener
import gym
from gym import spaces
from gym.envs.classic_control import rendering


# Some of parameters below are the same as in LunarLander-v2 (https://gym.openai.com/envs/LunarLander-v2/)
FPS = 60
SCALE = 30.0   # Affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6
FULL_TANK = 500   # Full tank of fuel

INITIAL_RANDOM_FORCE = 1000.0

LANDER_POLY =[(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]

LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
W = SCREEN_WIDTH / SCALE
H = SCREEN_HEIGHT / SCALE

WIND_SPEED_RANGE = np.arange(10, 100) / W / SCALE   # The wind speed is selecting in this range
WIND_SPEED_SHIFT = 10 / W / SCALE   # Gusts of wind

RELIEF_COLOR = np.array([222, 184, 135]) / 255       # burlywood
SKY_COLOR = np.array([255, 228, 196]) / 255          # bisque
LANDER_COLOR = np.array([176, 196, 222]) / 255       # lightsteelblue
LANDER_COLOR_BORDER = np.array([70, 130, 180]) / 255 # steelblue
LEGS_COLOR = np.array([192, 192, 192]) / 255         # silver
LEGS_COLOR_BORDER = np.array([119, 136, 153]) / 255  # lightslategray 



class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.terminated = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class MarsLanderEnvironment(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }
    continuous = False

    def __init__(self):
        self._seed()
        self.viewer = None

        self.world = Box2D.b2World()   #
        self.mars = None    #
        self.lander = None   #
        self.fire = []    # Particles representing fire

        self.prev_reward = None    #

        high = np.array([np.inf]*8)  # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-high, high)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,))
        else:
            # ACTIONS: no fire, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        self._reset()

    def _seed(self, seed=None):
        random.seed(seed)
        np.random.random(seed)
        return [seed]

    def _destroy(self):
        if not self.mars: 
            return
        self.world.contactListener = None
        self._clean_fire(True)
        self.world.DestroyBody(self.mars)
        self.mars = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def _reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.terminated = False
        self.prev_shaping = None
        self.human_render = False

        # Each episode we choose a new wind speed and the direction on the wind (from to the right and vise versa)
        self.wind_speed = random.choice(WIND_SPEED_RANGE)
        self.wind_direction = random.choice((-1, 1))    # -1 means from right to left; 1 means from left to right
        # Le't monitor the fuel capacity
        self.fuel_capacity = FULL_TANK

        self.total_angle = 0
        self.prev_angle = 0

        # Relief
        BENDS = 15
        bends_y = np.random.uniform(0, H/2, size=(BENDS+1,))
        hill_width = W/(BENDS-1)
        bends_x = [hill_width*i + np.random.uniform(-0.5*hill_width, 0.5*hill_width) * int(n != 0 and n != BENDS-1) for n,i in enumerate(range(BENDS))]
        self.plateau_x1 = bends_x[BENDS//2-1]
        self.plateau_x2 = bends_x[BENDS//2+1]
        self.plateau_center = (self.plateau_x2 + self.plateau_x1) / 2
        self.plateau_y = np.random.uniform(H/8, H/2)
        bends_y[BENDS//2-2: BENDS//2+3] = self.plateau_y
        smooth_y = [np.mean([bends_y[i-1], bends_y[i+0], bends_y[i+1]]) for i in range(BENDS)]

        self.mars = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.relief_coords = []
        for i in range(BENDS-1):
            p1 = (bends_x[i], smooth_y[i])
            p2 = (bends_x[i+1], smooth_y[i+1])
            self.mars.CreateEdgeFixture(
                vertices=[p1,p2],
                density=0,
                friction=0.1)
            self.relief_coords.append([p1, p2, (p2[0], 0), (p1[0], 0)])

        # Lander
        # Body
        initial_y = H
        initial_x = np.random.uniform(0.1*W, 0.9*W)
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x/SCALE, y/SCALE) for x,y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,   # collide only with ground
                restitution=0.0)    # 0.99 bouncy
                )
        self.lander.color1 = LANDER_COLOR
        self.lander.color2 = LANDER_COLOR_BORDER
        self.lander.ApplyForceToCenter((
            np.random.uniform(-INITIAL_RANDOM_FORCE, INITIAL_RANDOM_FORCE),
            np.random.uniform(-INITIAL_RANDOM_FORCE, INITIAL_RANDOM_FORCE)
            ), True)

        # Legs
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i*LEG_AWAY/SCALE, initial_y),
                angle=(i*0.05),
                fixtures = fixtureDef(
                    shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            leg.ground_contact = False
            leg.color1 = LEGS_COLOR
            leg.color2 = LEGS_COLOR_BORDER
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i*LEG_AWAY/SCALE, LEG_DOWN/SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3*i  # low enough not to jump back into the sky
                )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs
        return self._step(np.array([0, 0]) if self.continuous else 0)[0]

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x,y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2/SCALE, pos=(0,0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
                )
        p.ttl = ttl
        self.fire.append(p)
        self._clean_fire(False)
        return p

    def _clean_fire(self, all):
        while self.fire and (all or self.fire[0].ttl < 0):
            self.world.DestroyBody(self.fire.pop(0))

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid " % (action,type(action))

        # Engines
        tip = (np.sin(self.lander.angle), np.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [np.random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        # Wind speed is the average wind plus or minus a wind rush
        self.current_wind_speed = self.wind_direction * \
                                  (self.wind_speed + np.random.uniform(-WIND_SPEED_SHIFT, WIND_SPEED_SHIFT))
        self.lander.linearVelocity[0] += self.current_wind_speed

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0

            self.fuel_capacity -= m_power   # Update the fuel capacity
            
            ox = tip[0]*(4/SCALE + 2*dispersion[0]) + side[0]*dispersion[1]   # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1]*(4/SCALE + 2*dispersion[0]) - side[1]*dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)    # fire are just a decoration, 3.5 is here to make particle speed adequate
            p.ApplyLinearImpulse((ox*MAIN_ENGINE_POWER*m_power, oy*MAIN_ENGINE_POWER*m_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse((-ox*MAIN_ENGINE_POWER*m_power, -oy*MAIN_ENGINE_POWER*m_power), impulse_pos, True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1, 3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5,1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action-2
                s_power = 1.0
            
            self.fuel_capacity -= s_power * 0.1     # Update the fuel capacity

            ox = tip[0]*dispersion[0] + side[0]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
            oy = -tip[1]*dispersion[0] - side[1]*(3*dispersion[1]+direction*SIDE_ENGINE_AWAY/SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0]*17/SCALE, self.lander.position[1] + oy + tip[1]*SIDE_ENGINE_HEIGHT/SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse((ox*SIDE_ENGINE_POWER*s_power,  oy*SIDE_ENGINE_POWER*s_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse((-ox*SIDE_ENGINE_POWER*s_power, -oy*SIDE_ENGINE_POWER*s_power), impulse_pos, True)

        self.world.Step(1.0/FPS, 6*30, 2*30)

        #
        pos = self.lander.position
        vel = self.lander.linearVelocity
        if np.sign(self.prev_angle) != np.sign(self.lander.angle):
            self.total_angle = 0
        self.total_angle += self.lander.angle

        W2 = W / 2
        state = [
            (pos.x - self.plateau_center) / W2,
            (pos.y - (self.plateau_y + LEG_DOWN/SCALE)) / W2,
            vel.x*W2/FPS,
            vel.y*W2/FPS,
            self.lander.angle,
            20.0*self.lander.angularVelocity/FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
        ]
        assert len(state) == 8

        reward = 0
        shaping = \
            - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
            - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
            - 100*abs(state[4]) \
            + 10*state[6] + 10*state[7]   # And ten points for legs contact, the idea is if you lose
                                          # contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        #reward -= 10 * abs(self.total_angle)
        self.prev_shaping = shaping

        reward -= m_power + 0.1*s_power     # Less fuel spent is better

        done = False
        if self.terminated or abs(state[0]) >= 1.5:
            done = True
            reward = -100
        if not self.lander.awake:
            done = True
            reward = +100
        if self.fuel_capacity <= 0:     # Too many actions is bad because of limit of fuel
            done = True
            reward = -100

        wind_label = "Wind: {0:.1f} m/s".format(abs(self.current_wind_speed) * W * SCALE)
        fuel_label = "Fuel: {0:.1f}%".format(self.fuel_capacity / FULL_TANK * 100)

        return np.array(state), reward, done, \
               {"wind_label": wind_label, "fuel_label": fuel_label, 'x': pos.x, 'y': pos.y, 'angle': self.lander.angle}

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(SCREEN_WIDTH, SCREEN_HEIGHT)
            self.viewer.set_bounds(0, SCREEN_WIDTH/SCALE, 0, SCREEN_HEIGHT/SCALE)   #

        #  Draw fire
        for obj in self.fire:
            obj.ttl -= 0.15
            # Change fire particles color (RGB)
            particle_color = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))
            obj.color1 = particle_color
            obj.color2 = particle_color
        self._clean_fire(False)

        # Draw sky
        self.viewer.draw_polygon([(0, 0), (0, SCREEN_HEIGHT), (SCREEN_WIDTH, SCREEN_HEIGHT), (SCREEN_WIDTH, 0)],
                                 color=SKY_COLOR)
        # Draw relief
        for p in self.relief_coords:
            self.viewer.draw_polygon(p, color=RELIEF_COLOR)

        for obj in self.fire + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
        # Draw flags
        flagy1 = self.plateau_y
        flagy2 = flagy1 + 50 / SCALE
        if self.terminated:
            flag_color = (1.0, 0.0, 0.0)
        else:
            flag_color = (1.0, 1.0, 1.0)
        for x in [self.plateau_x1, self.plateau_x2]:
            # Pillars
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(0, 0, 0))
            # Flags
            self.viewer.draw_polygon(
                [(x + self.wind_direction / SCALE, flagy2), (x + self.wind_direction / SCALE, flagy2 - 10 / SCALE),
                 (x + self.wind_direction * 25 / SCALE, flagy2 - 5 / SCALE)],
                color=flag_color)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


if __name__=="__main__":
    env = MarsLanderEnvironment()
    s = env.reset()
    total_reward = 0
    steps = 0
    while True:
        a = random.choice([0])
        s, r, done, info = env.step(a)
        env.render()
        total_reward += r
        if steps % 20 == 0 or done:
            print(["{:+0.2f}".format(x) for x in s])
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done:
            break
    env.close()