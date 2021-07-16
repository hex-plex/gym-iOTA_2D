import gym 
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

import pygame
from pygame.locals import(QUIT,KEYDOWN,K_ESCAPE)

import Box2D
from Box2D.b2 import (world,polygonShape,circleShape,staticBody,dynamicBody)


class Iota2DEnv(gym.Env):
    metadata={'render.modes':['human']}

    def __init__(self):
        self.pixels_per_metre = 20,
        self.n =  10
        self.arena = (10.,10.)
        self.target_pos = (1,1)
        self.box_side = 1,
        self.robot_radius=0.1,
        high = np.array([(np.array(self.arena)) for _ in range(self.n)])

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
        
        self.box = self.world.CreateDynamicBody(position=(5,5))
        self.box_fixture = self.box.CreatePolygonFixture(
            box=(self.box_side,self.box_side),
            density=1,
            friction=0
            )
        
        positions = [(3*(i % 3),3*(i // 3)) for i in range(10)]

        self.robots = [
            self.world.CreateDynamicBody(position=position) for position in positions
        ]

        self.robots_fixtures = [
            robot.CreateCircleFixture(radius=self.robot_radius,density=1,friction=0) 
            for robot in self.robots
        ]

        self.screen_height = 480
        self.screen_width = 680
        self.fps = 60
        self.time_step = 1./self.fps

        colors = {
            'box': (255,255,255,255),
            'robot': (127,127,127, 255),
        }

        def draw_poly(polygon, body, fixture):
            vertices = [(body.transform * v) * self.pixels_per_metre for v in polygon.vertices]
            vertices = [(v[0], self.screen_height - v[1]) for v in vertices]
            pygame.draw.polygon(self.screen, colors[body.type], vertices)

        polygonShape.draw = draw_poly

        def draw_circle(circle, body, fixture):
            position = body.transform * circle.pos * self.pixels_per_metre
            position = (position[0], self.screen_height - position[1])
            pygame.draw.circle(self.screen, colors[body.type], [int(
            x) for x in position], int(circle.radius * self.pixels_per_metre))

        circleShape.draw = draw_circle

        print('__init__-works')

    def step(self,action):
        print('step-works')

    def reset(self):
        print('reset-works')

    def render(self, mode='human',close=False):
        print('render-works')
        