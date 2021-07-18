import gym 
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

import pygame
from pygame.locals import(QUIT,KEYDOWN,K_ESCAPE)

import Box2D
from Box2D.b2 import (world,polygonShape,circleShape,staticBody,dynamicBody,vec2)


class Iota2DEnv(gym.Env):
    metadata={'render.modes':['human']}

    def __init__(self):
        self.pixels_per_metre = 30
        self.n =  10
        self.arena = (10,10)
        self.target_pos = (0,5)
        self.box_side = 1
        self.robot_radius=0.3
        self.max_velocity = 1.
        self.max_force = 1.
        self.epsilon = 0.01
        high = np.array([(np.array(self.arena,dtype=np.float64)) for _ in range(self.n)])

        self.action_space = spaces.Box(-high,high,
        dtype=np.float64,
        shape=(self.n,2)    #sanity check
            )
        
        self.observation_space = spaces.Box(
            -high,
            high,
            dtype=np.float64,
            shape=(self.n,2)    #sanity check
        )

        # world setup
        self.screen=None
        self.world = world(gravity=(0,0))
        # TODO : positions
        self.box = self.world.CreateDynamicBody(position=(0,0),linearDamping=5,angularDamping=5)
        self.box_fixture = self.box.CreatePolygonFixture(
            box=(self.box_side,self.box_side),
            density=1,
            friction=0
            )
        xchoices =[]
        ychoices = []
        for i in range (-80,85,5):
            if np.abs(i)>20:
                xchoices.append(i/10.)
                ychoices.append(i/10.)
        np.random.shuffle(xchoices)
        np.random.shuffle(ychoices)
        print(list(zip(xchoices[:self.n],ychoices[:self.n])))
        positions = list(zip(xchoices[:self.n],ychoices[:self.n]))
        self.robots = [
            self.world.CreateDynamicBody(position=position) for position in positions
        ]

        self.robots_fixtures = [
            robot.CreateCircleFixture(radius=self.robot_radius,density=1,friction=0) 
            for robot in self.robots
        ]
        self.screen_width,self.screen_height = 2*np.array(self.arena)*self.pixels_per_metre
        self.fps = 60
        self.time_step = 1./self.fps

        colors = {
            'box': (255,255,255,255),
            'robot': (127,127,127, 255),
        }

        def draw_box(polygon, body, fixture):
            vertices = [(body.transform * v + self.arena) * self.pixels_per_metre for v in polygon.vertices]
            vertices = [(v[0], self.screen_height - v[1]) for v in vertices]
            pygame.draw.polygon(self.screen, colors['box'], vertices)

        polygonShape.draw = draw_box

        def draw_robot(circle, body, fixture):
            position = (body.transform * circle.pos + self.arena) * self.pixels_per_metre
            position = (position[0], self.screen_height - position[1])
            pygame.draw.circle(self.screen, colors['robot'], [int(
            x) for x in position], int(circle.radius * self.pixels_per_metre))

        circleShape.draw = draw_robot

        print('__init__-works')

    def step(self,action):
        err_msg = "%r (%s) invalid" % (action,type(action))
        assert self.action_space.contains(action), err_msg
        # TODO implement step 
        finished = False
        while not finished:
            finished =True
            for robot,destination in zip(self.robots,action):
                delta = vec2(destination) - robot.position
                if delta.length <= self.epsilon:
                    robot.linearVelocity = (0,0)
                    continue
                finished = False
                direction = delta/delta.length
                vel_mag = robot.linearVelocity.length * direction.dot(robot.linearVelocity)
                force_mag = self.max_force*(1 - vel_mag/self.max_velocity)
                robot.ApplyForce(force = force_mag*direction,point=robot.position,wake=True)

            self.world.Step(self.time_step,10,10)
            self.render() 
        self.world.Step(self.time_step,10,10) 
        observation = np.array([np.array(robot.position) for robot in self.robots])
        dx,dy = self.box.position - self.target_pos
        reward = -(dx**2 + dy**2)
        done = reward == 0
        return observation,reward, done, { }

    def reset(self):
        print(np.array(self.robots[0].position))
        print('reset-works')
        return np.array([np.array(robot.position) for robot in self.robots])

    def render(self, mode='human',close=False):
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.screen_width,self.screen_height),0,32)
            pygame.display.set_caption('iOTA-2D')
            self.clock = pygame.time.Clock()

        self.screen.fill((0,0,0,0))

        for body in self.world.bodies:
            for fixture in body.fixtures:
                fixture.shape.draw(body,fixture)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()
        