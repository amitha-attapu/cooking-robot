"""
    This code communicates with the coppeliaSim software and simulates shaking a container to mix objects of different color 

    Install dependencies:
    https://www.coppeliarobotics.com/helpFiles/en/zmqRemoteApiOverview.htm
    
    MacOS: coppeliaSim.app/Contents/MacOS/coppeliaSim -GzmqRemoteApi.rpcPort=23004 ~/path/to/file/mix_Intro_to_AI.ttt
    Ubuntu: ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23004 ~/path/to/file/mix_Intro_to_AI.ttt
"""

import random
import sys
# Change to the path of your ZMQ python API
sys.path.append('/app/zmq/')
import numpy as np
from zmqRemoteApi import RemoteAPIClient
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import itertools
import torch.nn.functional as F


class Simulation():
    def __init__(self, sim_port = 23000):
        self.sim_port = sim_port
        self.directions = ['Up','Down','Left','Right']
        self.initializeSim()

    def initializeSim(self):
        self.client = RemoteAPIClient('localhost',port=self.sim_port)
        self.client.setStepping(True)
        self.sim = self.client.getObject('sim')
        
        # When simulation is not running, ZMQ message handling could be a bit
        # slow, since the idle loop runs at 8 Hz by default. So let's make
        # sure that the idle loop runs at full speed for this program:
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)  
        
        self.getObjectHandles()
        self.sim.startSimulation()
        self.dropObjects()
        self.getObjectsInBoxHandles()
    
    def getObjectHandles(self):
        self.tableHandle=self.sim.getObject('/Table')
        self.boxHandle=self.sim.getObject('/Table/Box')
    
    def dropObjects(self):
        self.blocks = 18
        frictionCube=0.06
        frictionCup=0.8
        blockLength=0.016
        massOfBlock=14.375e-03
        
        self.scriptHandle = self.sim.getScript(self.sim.scripttype_childscript,self.tableHandle)
        self.client.step()
        retInts,retFloats,retStrings=self.sim.callScriptFunction('setNumberOfBlocks',self.scriptHandle,[self.blocks],[massOfBlock,blockLength,frictionCube,frictionCup],['cylinder'])
        
        print('Wait until blocks finish dropping')
        while True:
            self.client.step()
            signalValue=self.sim.getFloatSignal('toPython')
            if signalValue == 99:
                loop = 20
                while loop > 0:
                    self.client.step()
                    loop -= 1
                break

    # def test_agent():
    #     load the q table
    #     for episode in range(200):
    #         for _ in range(steps):
    #             for that particular states, check q table what action to take
    
    
    def getObjectsInBoxHandles(self):
        self.object_shapes_handles=[]
        self.obj_type = "Cylinder"
        for obj_idx in range(self.blocks):
            obj_handle = self.sim.getObjectHandle(f'{self.obj_type}{obj_idx}')
            self.object_shapes_handles.append(obj_handle)

    def getObjectsPositions(self):
        pos_step = []
        box_position = self.sim.getObjectPosition(self.boxHandle,self.sim.handle_world)
        for obj_handle in self.object_shapes_handles:
            # get the starting position of source
            obj_position = self.sim.getObjectPosition(obj_handle,self.sim.handle_world)
            obj_position = np.array(obj_position)
            pos_step.append(list(obj_position[:2]))
        return pos_step
    
    def getBoxPosition(self):
        return self.sim.getObjectPosition(self.boxHandle,self.sim.handle_world)
    
    def action(self,direction=None):
        if direction not in self.directions:
            print(f'Direction: {direction} invalid, please choose one from {self.directions}')
            return
        box_position = self.sim.getObjectPosition(self.boxHandle,self.sim.handle_world)
        _box_position = box_position
        span = 0.02
        steps = 5
        if direction == 'Up':
            idx = 1
            dirs = [1, -1]
        elif direction == 'Down':
            idx = 1
            dirs = [-1, 1]
        elif direction == 'Right':
            idx = 0
            dirs = [1, -1]
        elif direction == 'Left':
            idx = 0
            dirs = [-1, 1]

        for _dir in dirs:
            for _ in range(steps):
                _box_position[idx] += _dir*span / steps
                self.sim.setObjectPosition(self.boxHandle, self.sim.handle_world, _box_position)
                self.stepSim()
        
        

    def getDirectionNo(self,direction):
        if(direction=='Up'):
            return 0
        if(direction=='Down'):
            return 1
        if(direction=='Right'):
            return 2
        if(direction=='Left'):
            return 3
        
    def getDirection(self,directionNo):
        if(directionNo==0):
            return 'Up'
        if(directionNo==1):
            return 'Down'
        if(directionNo==2):
            return 'Right'
        if(directionNo==3):
            return 'Left'

    def stepSim(self):
        self.client.step()

    def stopSim(self):
        self.sim.stopSimulation()

def get_current_state(env):
    box_position = env.getBoxPosition()
    positions = env.getObjectsPositions()
    first,second,third,fourth = [0,0] , [0,0] , [0,0] ,[0,0]
    for i in range(18):
        #print("box position : ",box_position," cylinder positions : ",positions[i])
        if(positions[i][0]<box_position[0] and positions[i][1]<box_position[1]):
            if i<9:
                third[0]=third[0]+1
            else:
                third[1]=third[1]+1
        if(positions[i][0]<box_position[0] and positions[i][1]>box_position[1]):
            if i<9:
                first[0]=first[0]+1
            else:
                first[1]=first[1]+1
        if(positions[i][0]>box_position[0] and positions[i][1]<box_position[1]):
            if i<9:
                fourth[0]=fourth[0]+1
            else:
                fourth[1]=fourth[1]+1
        if(positions[i][0]>box_position[0] and positions[i][1]>box_position[1]):
            if i<9:
                second[0]=second[0]+1
            else:
                second[1]=second[1]+1
    state=0
    reward=0
    if(first[0]==first[1]):
        state+=1
        reward+=1
    else:
        reward-=1
    if(second[0]==second[1]):
        state+=2
        reward+=1
    else:
        reward-=1
    if(third[0]==third[1]):
        state+=4
        reward+=1
    else:
        reward-=1
    if(fourth[0]==fourth[1]):
        state+=8
        reward+=1
    else:
        reward-=1
    
    # if(state == 7 or state ==  11 or state == 13 or  state == 14 or state ==  3 or state == 5 or  state == 6 or  state == 9 or  state == 10 or  state == 12 or  state == 15):
    #     reward+=2
    # if(state == 15):
    #     print("current state is: ", state," !!!!!", first,second,third,fourth)
    # print("current state is: ", state," !!!!!", first,second,third,fourth)
    return state,reward
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 4)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    def act(self, obs):
        bin_obs = bin(obs)
        # print("bin obs ",bin_obs,'{0:04b}'.format(obs),obs)
        modified_list=[int(i) for  element in '{0:04b}'.format(obs) for i in element]
        # print("modified_list ",modified_list)
        obs_t = torch.as_tensor(modified_list, dtype=torch.float32)
        # print("obs :" , obs_t)
        # print("here")
        q_values = self(obs_t.unsqueeze(0))
        # print("q_values : ", q_values)
        max_q_index = torch.argmax(q_values)
        direction = max_q_index.detach().item()
        # print("direction: ",direction)
        return direction
    
def test_agent():
    online_net = Network()
    online_net.load_state_dict(torch.load("model"))
    for episode in range(100):
        env= Simulation()
        current_state,reward = get_current_state(env)
        for step in range(30):
            if(current_state==15):
                print("reached final state, episode: ",episode)
                break
            directionNo = online_net.act(current_state)
            env.action(direction = env.getDirection(directionNo))
            current_state,reward = get_current_state(env)
        env.stopSim()
 

    
def main():
    test_agent()


if __name__ == '__main__':
    
    main()
