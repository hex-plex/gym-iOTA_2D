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

    def __init__(self,no_of_modules=10, arena=(5,5),pixels_per_metre=60,
    box_side=0.5,robot_radius=0.1,max_velocity=2.,max_force=2.,epsilon=None,
    step_fps_ratio=10,robot_friction_coefficient=0.3,box_friction_coefficient=0.3,
    gravity=10,target_pos=None):
        self.ppm = pixels_per_metre
        self.n =  no_of_modules
        self.arena = arena
        self.target_pos = target_pos if target_pos is not None else (-arena[0],0)
        self.box_side = box_side
        self.robot_radius=robot_radius
        self.max_velocity = max_velocity
        self.max_force = max_force
        self.epsilon = epsilon if epsilon is not None else 0.1 * robot_radius
        self.sfr = step_fps_ratio
        self.rfc = robot_friction_coefficient
        self.bfc = box_friction_coefficient
        self.gravity = gravity
        self.fps = 60
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

        self.screen_width,self.screen_height = 2*np.array(self.arena)*self.ppm
        self.time_step = 1./(self.fps*self.sfr)

        colors = {
            'box': (255,255,255,255),
            'robot': (127,127,127, 255),
        }

        def draw_box(polygon, body, fixture):
            vertices = [(body.transform * v + self.arena) * self.ppm for v in polygon.vertices]
            vertices = [(v[0], self.screen_height - v[1]) for v in vertices]
            pygame.draw.polygon(self.screen, colors['box'], vertices)

        polygonShape.draw = draw_box

        def draw_robot(circle, body, fixture):
            position = (body.transform * circle.pos + self.arena) * self.ppm
            position = (position[0], self.screen_height - position[1])
            pygame.draw.circle(self.screen, colors['robot'], [int(
            x) for x in position], int(circle.radius * self.ppm))

        circleShape.draw = draw_robot

        print('__init__-works')

    def step(self,action):
        err_msg = "%r (%s) invalid" % (action,type(action))
        assert self.action_space.contains(action), err_msg 
        finished = False
        steps=0
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
                force = force_mag*direction
                if robot.linearVelocity.length!=0:
                    force-= self.robot_friction * robot.linearVelocity/robot.linearVelocity.length
                robot.ApplyForce(force = force,point=robot.position,wake=True)
                
            if self.box.linearVelocity.length != 0:
                self.box.ApplyForceToCenter(
                    force=-self.box_friction*self.box.linearVelocity/self.box.linearVelocity.length,
                    wake=True)        

            self.world.Step(self.time_step,10,10)
            if steps%self.sfr == 0:
                self.render() 
            steps = (steps+1)
            if steps*self.time_step > 30:
                raise RuntimeError("environment timestep exceeded 30 seconds")

        self.world.Step(self.time_step,10,10)
        self.world.ClearForces()
        observation = np.array([np.array(robot.position) for robot in self.robots])
        target_delta = self.box.position - self.target_pos
        done = target_delta.length<= self.epsilon
        return observation,-target_delta.lengthSquared, done, { }

    def reset(self):
        self.screen=None
        self.world = world(gravity=(0,0))
        # TODO : positions
        self.box = self.world.CreateDynamicBody(position=(0,0),angularDamping=5)
        self.box_fixture = self.box.CreatePolygonFixture(
            box=(self.box_side,self.box_side),
            density=1,
            friction=0
            )
        self.min_separation = 2*self.robot_radius + self.epsilon
        self.box_friction = self.box.mass*self.gravity*self.bfc
        
        positions =[]
        while len(positions) != self.n:
            pot_pos = vec2((
                np.random.uniform(low=-self.arena[0],high=self.arena[0]),
                np.random.uniform(low=-self.arena[1],high=self.arena[1])
                ))
            if all((pot_pos - position).length>=self.min_separation for position in positions):
                positions.append(pot_pos)

        self.robots = [
            self.world.CreateDynamicBody(position=position) for position in positions
        ]

        self.robots_fixtures = [
            robot.CreateCircleFixture(radius=self.robot_radius,density=1,friction=0) 
            for robot in self.robots
        ]

        self.robot_friction = self.robots[0].mass*self.gravity*self.rfc

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
        