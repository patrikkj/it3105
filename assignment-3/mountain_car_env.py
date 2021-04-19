import math
import random
import numpy as np

class MountainCar():
    REWARD_WIN = 0
    REWARD_ACTION = -1

    def __init__(self, x_range=[-1.2,0.6], v_range= [-0.07,0.07], max_steps=1000):
        self.actions = [-1,0,1]
        self.x_range = x_range
        self.v_range = v_range
        self.x = self.initial_x()
        self.v = 0
        self.max_steps = max_steps
        self.step = 0
        self.last_action = 0
        

    def apply_action(self, action):
        #print("----------------------------------------------------------")
        #print("ACTION:", action, "      (from mc.apply_action())")
        #print("----------------------------------------------------------")

        self.v = self.next_v(self.x, self.v, action)
        self.x = self.next_x(self.x, self.v)
        self.step +=1
        self.last_action = action
        reward = MountainCar.REWARD_ACTION
        if self.is_completed: 
            reward = MountainCar.REWARD_WIN
        #print("Step: ", self.step, "  of " , self.max_steps)
        return self.get_observation(), reward, self.is_finished()

    def get_observation(self):
        return (self.x, self.v)

    def next_v(self, x, v, action):
        next_v =  v + 0.001 * int(action) - 0.0025 * math.cos(3 * x)
        if next_v > 0:
            return min(next_v, self.v_range[1])
        else: 
            return max(next_v, self.v_range[0])

    def next_x(self, x, v):
        next_x =  x + v
        if next_x > 0 :
            return min(next_x, self.x_range[1])
        else: 
            return max(next_x, self.x_range[0])

    def is_finished(self):
        return self.is_timeout() or self.is_completed()

    def is_timeout(self):
        return self.step >= self.max_steps
    
    def is_completed(self):
        return self.x == self.x_range[1]
    
    def visualize(self):
        #funcAnimation
        print("x: ", self.x, "  ,   v: ", self. v)#, "  , Action: , self.last_action, "\n")

    def initial_x(self):
        return random.uniform(-0.6,-0.4)
    
    def random_action(self):
        return random.choice(self.actions)

    def naive_action(self):
        if self.v > 0:
            return self.actions[2]
        elif self.v == 0:
            return self.actions[1]
        else:
            return self.actions[0]
    
    def reset(self):
        self.x = self.initial_x()
        self.v = 0
        self.step = 0
        self.last_action = 0
    
    def get_legal_actions(self):
        return self.actions



    #--------- Tiling methods -------------


    def init_tilings(self, x_range=[0,0], v_range=[0,0], n_tiles=0, n_tilings=0, displacement_vector=[0,0]):
        self.displacement_vector = displacement_vector
        self.n_tiles = n_tiles
        self.n_tilings = n_tilings
        self.displacement_vector= np.array(displacement_vector)

        # List of displacement vectors indexed by tiling number --> [(1,3), (2,6), (4,9) ...] (example for asymmetrical displacement (3,1))
        self.tiling_displacement = np.array([self.displacement_vector* i for i in range(self.n_tilings)])
        # List of tile widths in each dimension --> [0.3 , 0,25]
        self.tile_width = np.array([(self.x_range[1]-self.x_range[0])/self.n_tiles , (self.v_range[1]-self.v_range[0])/self.n_tiles])
        # The offset between tilings --> [0.02, 0.045]
        self.offset = self.tile_width / (self.n_tilings-1)
        self.extra_tiles = np.array([math.ceil(self.offset[k] * self.tiling_displacement[len(self.tiling_displacement)-1][k] / self.tile_width[k]) for k in range(len(self.offset)) ]) 
        self.total_tiles = self.n_tiles + self.extra_tiles
        print ("-----------------------------------------------------")
        
        print("v tile width: ", (self.v_range[1]-self.v_range[0])/self.n_tiles)
        print("n_tiles: ", self.n_tiles)
        print("Extra tiles needed: ", self.extra_tiles)
        print ("total tiles: " , self.total_tiles)
        print("n_tilings: ", self.n_tilings)
        print("Tile width:" , self.tile_width)
        print("Tiling displacement:" , self.tiling_displacement)
        print("offset: ", self.offset)
        print ("-----------------------------------------------------")

    def decode_state(self, x, v):
        """
        Finds which tile the (x, v) coordinate is in for each tiling and represents the state as an integer corresponding to a binary string.

        Principles:
        * x // tile_width indicate which tile x is in (0-indexed)
            Each tiling is offset in the direction of up and to the right
            To account for the fact that each tiling is offset, the offset in the given dimension times the displacement of that tiling is subtracted. 
            To account for the fact that the bottom left corner is not origo, the start range value is subtracted
            Finally, accounting for extra tiles added is added. Extra tiles are there to ensure that all feasible points are within each tiling even after offsetting 
        
        * Each state is represented by an element in a state vector
            Each state vector element corresponds to one tile in one tiling
            The state vector represents the different tiles like this: [t^1_(1,1) , t^1_(1,2) ... t^(1)_(n_tiles,n_tiles) , t^(2)_(1,1) ... ... t^(n_tilings)_(n_tiles,n_tiles)]

            If the (x, v) coordinate is in a certain tile for a given tiling, the corresponding element in the state vector will be 1
        """
    

        #print(self.offset[0] * self.tiling_displacement[len(self.tiling_displacement)-1][0] / self.tile_width[0])

        #state = 0
        n_features = np.product(self.total_tiles) * self.n_tilings
        state = np.zeros(n_features, dtype=int)

        for i in range(self.n_tilings):
            # Finds the index of the tile in both dimensions
            x_tile = (x   -    self.offset[0] * self.tiling_displacement[i][0]    -    self.x_range[0]   +    self.extra_tiles[0]  *   self.tile_width[0])     //    self.tile_width[0]
            v_tile = (v   -    self.offset[1] * self.tiling_displacement[i][1]    -    self.v_range[0]   +    self.extra_tiles[1]   *   self.tile_width[1])      //    self.tile_width[1]
            
            #x_tile = (x   -    self.offset[0] * self.tiling_displacement[i][0]    -    self.x_range[0]   +    self.extra_tiles[0]  *   self.tile_width[0])     //    self.tile_width[0]
            #v_tile = (v   -    self.offset[1] * self.tiling_displacement[i][1]    -    self.v_range[0]   +    self.extra_tiles[1]   *   self.tile_width[1])      //    self.tile_width[1]

            index = int(i * (self.total_tiles[0]*self.total_tiles[1]) + x_tile * self.total_tiles[0] + v_tile)
            #print("INDEX" , index)
            state[index] = 1

            """
            # adds the correct bit (corresponding to the state of the tiling) to the state integer
            state += 2 ** (i * self.n_tiles**2 + x_tile * self.n_tiles + v_tile)
            """
            #print ("Tiling %s: (%s,%s)" % (i, x_tile, v_tile))

        return tuple(state)