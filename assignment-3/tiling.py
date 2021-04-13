import math
import numpy as np
import random


class Tiling():

    def __init__(self, x_range=[-1.2,0.6], v_range= [-0.07,0.07], n_tiles=4, n_tilings=5, displacement_vector=[1,3]):
        
        self.x_range = np.array(x_range)
        self.v_range = np.array(v_range)
        self.x = 0
        self.v = 0
        self.last_action = 0

        self.n_tiles = n_tiles              # nxn tiles in a tiling (i.e. n_tiles=4 --> each tiling has 4x4 tiles)
        self.n_tilings = n_tilings          # Number of tilings (grids) overlayed with different offsets
        self.displacement = np.array(displacement_vector)

        self.init_tilings()
        
        

    def init_tilings(self):
        # List of displacement vectors indexed by tiling number --> [(1,3), (2,6), (4,9) ...] (example for asymmetrical displacement (3,1))
        self.tiling_displacement = np.array([self.displacement * i for i in range(self.n_tilings)])
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
        

    def convert_state(self, x, v):
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

        IGNORE 
        * Each state can be represented as a binary string where each bit corresponds to one tile in one tiling. 
            The bitstring represents a state vector where each tile is represented like this: [t^1_(1,1) , t^1_(1,2) ... t^(1)_(n_tiles,n_tiles) , t^(2)_(1,1) ... ... t^(n_tilings)_(n_tiles,n_tiles)]
            Each bit corresponds to an element in this state vector (the vector itself is not created)
            The bitstring is returned as an integer
        """
    

        #print(self.offset[0] * self.tiling_displacement[len(self.tiling_displacement)-1][0] / self.tile_width[0])

        #state = 0
        n_features = self.total_tiles[0] * self.total_tiles[1] * self.n_tilings
        state = np.zeros(n_features, dtype=int)
        print(np.shape(state))

        for i in range(self.n_tilings):
            # Finds the index of the tile in both dimensions
            x_tile = (x   -    self.offset[0] * self.tiling_displacement[i][0]    -    self.x_range[0]   +    self.extra_tiles[0]  *   self.tile_width[0])     //    self.tile_width[0]
            v_tile = (v   -    self.offset[1] * self.tiling_displacement[i][1]    -    self.v_range[0]   +    self.extra_tiles[1]   *   self.tile_width[1])      //    self.tile_width[1]
            
            #x_tile = (x   -    self.offset[0] * self.tiling_displacement[i][0]    -    self.x_range[0]   +    self.extra_tiles[0]  *   self.tile_width[0])     //    self.tile_width[0]
            #v_tile = (v   -    self.offset[1] * self.tiling_displacement[i][1]    -    self.v_range[0]   +    self.extra_tiles[1]   *   self.tile_width[1])      //    self.tile_width[1]

            index = int(i * (self.total_tiles[0]*self.total_tiles[1]) + x_tile * self.total_tiles[0] + v_tile)
            print("INDEX" , index)
            state[index] = 1




            """
            # adds the correct bit (corresponding to the state of the tiling) to the state integer
            state += 2 ** (i * self.n_tiles**2 + x_tile * self.n_tiles + v_tile)
            """
            print ("Tiling %s: (%s,%s)" % (i, x_tile, v_tile))

        return state 


def main():
    tiling = Tiling()
    #tiling = Tiling([-5,5],[-1,1],4,5,[1,3])
    state = tiling.convert_state(0.1 , 0.05)
    #print(state)

main()


    


