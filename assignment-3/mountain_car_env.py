import math
import random
import numpy as np

class MountainCar():
    REWARD_WIN = 0
    REWARD_ACTION = -1

    def __init__(self, x_range=[-1.2,0.6], v_range= [-0.07,0.07], max_steps=1000):
        self.actions = [-1,0,1]
        self.x_range = x_range
        print("INITINITIINIT x_range:", self.x_range)
        self.v_range = v_range
        self.x = self.initial_x()
        self.v = 0
        self.best_x = -10
        self.best_step = 0
        self.max_steps = max_steps
        self.step = 0
        self.last_action = 0
        self.figure_offset = [0,0]

    def apply_action(self, action):
        #print("----------------------------------------------------------")
        #print("ACTION:", action, "      (from mc.apply_action())")
        #print("----------------------------------------------------------")

        self.next_v(action)
        self.next_x()
        self.step +=1
        self.last_action = action
        reward = MountainCar.REWARD_ACTION
        if self.is_completed(): 
            print("ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ")
            print("ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ")
            print("ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ")
            print("ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ")
            print(f"(x,v): ({self.x},{self.v}, range: {self.x_range}")
            print("\n\n\n\n\n\n\\n\n\n\n\n\n\n")
            print("ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ")
            print("ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ")
            print("ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ")
            print("ÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆÆ")
            reward = MountainCar.REWARD_WIN
        #print("Step: ", self.step, "  of " , self.max_steps)
        self.check_best()
        return self.get_observation(), reward, self.is_finished()

    def get_observation(self):
        return (self.x, self.v)

    def next_v(self, action):
        next_v =  self.v + 0.001 * int(action) - 0.0025 * math.cos(3 * self.x)
        if next_v > 0:
            self.v = min(next_v, self.v_range[1])
        else: 
            self.v =  max(next_v, self.v_range[0])

    def next_x(self):
        next_x =  self.x + self.v
        if next_x > 0 :
            self.x = min(next_x, self.x_range[1])
        else: 
            self.x = max(next_x, self.x_range[0])

    def is_finished(self):
        return self.is_timeout() or self.is_completed()

    def is_timeout(self):
        return self.step >= self.max_steps
    
    def is_completed(self):
        return self.x >= self.x_range[1]
    
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
        self.best_x = -1000
        self.best_step = 0
    
    def get_legal_actions(self):
        return self.actions

    def check_best(self):
        if self.x > self.best_x:
            self.best_x = self.x
            self.best_step = self.step
        return

    #--------- Tiling methods -------------


    def init_tilings(self, n_tiles=[0,0], n_tilings=0, displacement_vector=[0,0]):
        self.displacement_vector = displacement_vector
        self.n_tiles = n_tiles
        self.n_tilings = n_tilings
        self.displacement_vector= np.array(displacement_vector)

        # List of displacement vectors indexed by tiling number --> [(1,3), (2,6), (4,9) ...] (example for asymmetrical displacement (3,1))
        self.tiling_displacement = np.array([self.displacement_vector* i for i in range(self.n_tilings)])
        # List of tile widths in each dimension --> [0.3 , 0,25]
        self.tile_width = np.array([(self.x_range[1]-self.x_range[0])/self.n_tiles[0] , (self.v_range[1]-self.v_range[0])/self.n_tiles[1]])
        # The offset between tilings --> [0.02, 0.045]
        
        #TESTING
        #self.offset = self.tile_width / (self.n_tilings)

        self.offset = self.tile_width / (self.n_tilings-1)
        self.extra_tiles = np.array([math.ceil(self.offset[k] * self.tiling_displacement[len(self.tiling_displacement)-1][k] / self.tile_width[k]) for k in range(len(self.offset)) ]) 
        self.total_tiles = self.n_tiles + self.extra_tiles
        self.start_coord = np.array([self.x_range[0],self.v_range[0]]) - self.extra_tiles*self.tile_width
        self.print_tiling_info()



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
        n_features = np.prod(self.total_tiles) * self.n_tilings
        state = np.zeros(n_features, dtype=int)

        for i in range(self.n_tilings):
            
            #TESTING
            x_tile = (x     -    self.start_coord[0]   -    self.offset[0] * self.tiling_displacement[i][0])       //    self.tile_width[0]
            v_tile = (v     -    self.start_coord[1]   -    self.offset[1] * self.tiling_displacement[i][1])       //    self.tile_width[1]
            
            
            
            
            # Finds the index of the tile in both dimensions
            #x_tile = (x     -    self.x_range[0]   -    self.offset[0] * self.tiling_displacement[i][0]     +    self.extra_tiles[0]  *   self.tile_width[0])     //    self.tile_width[0]
            #v_tile = (v     -    self.v_range[0]   -    self.offset[1] * self.tiling_displacement[i][1]     +    self.extra_tiles[1]   *   self.tile_width[1])      //    self.tile_width[1]
            
            
            #print(f"(x,v) = ({x},{v})       Tiling {i} , x_tile: {x_tile}  , v_tile: {v_tile}")


            #TESTING
            index = int(i * (self.total_tiles[0] * self.total_tiles[1]) + v_tile * self.total_tiles[0] + x_tile)




            #index = int(i * (self.total_tiles[0] * self.total_tiles[1]) + x_tile * self.total_tiles[0] + v_tile)
            #print("index: ", index)
            if index > n_features:
                self.print_tiling_info()
                print(" #############")
                print("x_tile: ", x_tile)
                print("v_tile: ", v_tile)
                print("index: ", index)
                print(self.total_tiles)
            #print("INDEX" , index)
            state[index] = 1

            #print ("Tiling %s: (%s,%s)" % (i, x_tile, v_tile))

        return tuple(state)


    def print_tiling_info(self):
        print ("-----------------------------------------------------")
        print("x tile width: ", (self.x_range[1]-self.x_range[0])/self.n_tiles[0])
        print("v tile width: ", (self.v_range[1]-self.v_range[0])/self.n_tiles[1])
        print("n_tiles: ", self.n_tiles)
        print("Extra tiles needed: ", self.extra_tiles)
        print ("total tiles: " , self.total_tiles)
        print("n_tilings: ", self.n_tilings)
        print("Start coord:", self.start_coord)
        print("Tile width:" , self.tile_width)
        print("Tiling displacement:" , self.tiling_displacement)
        print("offset: ", self.offset)
        print ("-----------------------------------------------------")

                