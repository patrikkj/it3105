import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import math
from utils import Direction
import time
import numpy as np

D = Direction
TRIANGLE_DIRECTIONS = [D.UP_LEFT, D.UP, D.LEFT, D.RIGHT, D.DOWN, D.DOWN_RIGHT]
DIAMOND_DIRECTIONS = [D.UP_RIGHT, D.UP, D.LEFT, D.RIGHT, D.DOWN, D.DOWN_LEFT]
# Node sizes
SMALL = 40
NORMAL = 150
LARGE = 400

node_colors = {0 : 'black', 1: "blue"}
rotation = {"triangle": -1/2 * math.pi, "diamond" : -1/4 * math.pi}
    #3/4
    # -7/12 for spring triangle

class Graphics():

    def __init__(self):
        self.directions = []
        self.game = []
        self.delay = 0.5
        self.board_type = "diamond"
        self.board_size = 0
        self.initialized = False
        self.G = nx.Graph()
        #self.testGame()
        self.step = 0
    
    def testGame(self):
            
            self.game = [{"board" : [1,1,1,1,1,1,1,1,1,1,1,    1,1,1,1,1,1,1,1,1,1,1,    1,1,0,1,1,1,1,1,1,1,1,   1,1,1,1,1,1,1,1,1,1,1,    1,1,1,1,1,1,1,1,1,1,1,    1,1,1,1,1,1,1,1,1,1,1,     1,1,1,1,1,1,1,1,1,1,1,       1,1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1,1,     1,1,1,1,1,1,1,1,1,1,1,       1,1,1,1,1,1,1,1,1,1,1], "peg_start_position": None, "peg_end_position" : None},
                        {"board" : [0,1,1,1,1,1,1,1,1,1,1,    1,0,1,1,1,1,1,1,1,1,1,    1,1,1,1,1,1,1,1,1,1,1,    1,1,1,1,1,1,1,1,1,1,1,    1,1,1,1,1,1,1,1,1,1,1,  1,1,1,1,1,1,1,1,1,1,1,     1,1,1,1,1,1,1,1,1,1,1,       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,     1,1,1,1,1,1,1,1,1,1,1,       1,1,1,1,1,1,1,1,1,1,1], "peg_start_position" : (0,0) , "peg_end_position" : (2,2)},
                        {"board" : [0,1,1,1,1,1,1,1,1,1,1,    1,1,1,1,1,1,1,1,1,1,1,    1,0,1,1,1, 1,1,1,1,1,1,   1,0,1,1,1,1,1,1,1,1,1,    1,1,1,1,1,1,1,1,1,1,1,  1,1,1,1,1,1,1,1,1,1,1,     1,1,1,1,1,1,1,1,1,1,1,       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,     1,1,1,1,1,1,1,1,1,1,1,       1,1,1,1,1,1,1,1,1,1,1], "peg_start_position": (3,1), "peg_end_position" : (1,1)}]
            """
            self.game = [{"board" : [1,1,1,1,1, 1,1,1,1,1,  1,1,0,1,1,  1,1,1,1,1,  1,1,1,1,1], "peg_start_position": None, "peg_end_position" : None},
                        {"board" : [0,1,1,1,1,  1,0,1,1,1,  1,1,1,1,1,  1,1,1,1,1,  1,1,1,1,1], "peg_start_position" : (0,0) , "peg_end_position" : (2,2)},
                        {"board" : [0,1,1,1,1,  1,1,1,1,1,  1,0,1,1,1,  1,0,1,1,1,  1,1,1,1,1], "peg_start_position": (3,1), "peg_end_position" : (1,1)},
                        {"board" : [0,1,1,1,1,  1,1,1,1,1,  1,1,1,1,1,  1,0,0,1,1,  1,1,1,0,1], "peg_start_position": (4,3), "peg_end_position" : (2,1)},
                        {"board" : [0,1,1,1,1,  1,1,1,1,1,  1,1,1,1,1,  1,0,0,1,1,  1,0,0,1,1], "peg_start_position": (4,1), "peg_end_position" : (4,3)},
                        {"board" : [0,1,1,1,1,  1,0,1,1,1,  1,0,1,1,1,  1,1,0,1,1,  1,0,0,1,1], "peg_start_position": (1,1), "peg_end_position" : (3,1)},
                        {"board" : [1,1,1,1,1,  0,0,1,1,1,  0,0,1,1,1,  1,1,0,1,1,  1,0,0,1,1], "peg_start_position": (2,0), "peg_end_position" : (0,0)},
                        {"board" : [1,1,1,1,1,  0,1,1,1,1,  0,0,0,1,1,  1,1,0,0,1,  1,0,0,1,1], "peg_start_position": (3,3), "peg_end_position" : (1,1)},
                        {"board" : [0,1,1,1,1,  0,0,1,1,1,  0,0,1,1,1,  1,1,0,0,1,  1,0,0,1,1], "peg_start_position": (0,0), "peg_end_position" : (2,2)}]
                
            """
    def visualize_episode(self, step_logs, board_type, episode, delay):
        """Initializes and runs visualize_game on correct episode. Called by other classes."""
        plt.ion()                                                                                        #interactive mode pyplot for updating graph plot
        self.delay = delay
        self.board_type = board_type
        self.directions = TRIANGLE_DIRECTIONS if self.board_type == "triangle" else DIAMOND_DIRECTIONS
        
        # Filters step_logs to correct episode and converts to dict
        self.game = step_logs[step_logs["episode"] == episode]
        print(self.game)
        self.game = self.game.to_dict(orient="records")

        # Convert from numpy types
        for step in self.game:
            step["board"] = (np.concatenate( step["board"], axis=0)).tolist()
            step["peg_start_position"] = tuple(step["peg_start_position"])
            step["peg_end_position"] = tuple(step["peg_end_position"])
            step["peg_move_direction"] = tuple(step["peg_move_direction"])

        self.visualize_game()
    
    def visualize_game(self):
        """Loops through the step_states of the chosen episode and visualizes each step_state. """
        for step_state in self.game:
            if not self.initialized:
                self.initialize(step_state)
                self.initialized = True 
            else:
                self.visualize(step_state)

            self.step += 1

    def visualize(self, step_state):
        # Update value for cells that have changed, i.e. the ones involved in the peg move
        start_pos  = step_state["peg_start_position"]
        end_pos = step_state["peg_end_position"]
        middle_pos = (start_pos[0] + 1/2 * (end_pos[0] - start_pos[0]) , start_pos[1] + 1/2 * (end_pos[1] - start_pos[1]))
    
        nx.set_node_attributes(self.G, {start_pos:  {'value': 0}})
        nx.set_node_attributes(self.G, {end_pos:  {'value': 1}})
        nx.set_node_attributes(self.G, {middle_pos:  {'value': 0}})

        #Update attributes of nodes the peg moved to and from
        if start_pos is not None:
            # Reset large and small nodes for last step's peg move
            nx.set_node_attributes(self.G, {self.game[self.step-1]["peg_start_position"] :  {'node_size': NORMAL}})
            nx.set_node_attributes(self.G, {self.game[self.step-1]["peg_end_position"] :  {'node_size': NORMAL}})

            # Reset edge color to black for all edges
            nx.set_edge_attributes(self.G, 'black', 'edge_color' )

            # Update hole size and color of the three nodes involved in a move
            nx.set_node_attributes(self.G, {start_pos :  {'node_size': SMALL, 'node_color': node_colors[0]}})
            nx.set_node_attributes(self.G, {end_pos :  {'node_size': LARGE, 'node_color': node_colors[1]}})
            nx.set_node_attributes(self.G, {middle_pos :  {'node_color': node_colors[0]}})
            
            #Update edge color of edges along peg move path
            nx.set_edge_attributes(self.G,{(start_pos, middle_pos)  : {'edge_color': 'red'}})
            nx.set_edge_attributes(self.G,{(end_pos, middle_pos)  : {'edge_color': 'red'}})
        
        # final is a boolean that determines whether it's the last step. This is so that the last "frame" will hold for more than self.delay seconds
        final = self.step == len(self.game) - 1
        self.show_graph(final)

    def show_graph(self, final):
        """Plots the graph with nodes and edges and pauses for self.delay time."""
        color_list = []
        size_list = []
        first_pos_list = {}
        pos_dict = {}
        degree = rotation[self.board_type]
        #degree = 0
        # Collect lists of node colors and node_sizes that will be arguments in the nx.draw_networkx function
        for node in self.G.nodes(data=True):
            #print("node: ", node)
            color_list.append(node[1]["node_color"])
            size_list.append(node[1]["node_size"])
            first_pos_list[node[0]] = node[0]
        
        # Edges         
        edges = self.G.edges()
        edge_colors = [self.G[u][v]["edge_color"] for u,v in edges]

        # Dict of positions of each node with node index as key. Rotated and sheared (if triangle) coordinates that will be for the pos argument in nx.draw_networkx function
        for pos in first_pos_list:
            row, col = self.rotate(pos,degree)
            pos_dict[pos] = (row,col)

        fig = plt.figure("Peg Solitaire")
        nx.draw_networkx(self.G, pos=pos_dict, with_labels= False, node_color = color_list, node_size = size_list, edge_color = edge_colors)
        if final:
            plt.pause(1000000)
        else:
            plt.pause(self.delay)
        
        plt.clf()

    def rotate(self, pos, degree):
        """Rotates and shears the position coordinates of the nodes in the graph depending on the board type."""
        pos0 = pos[0]
        pos1 = pos[1]
        
        # Rotation (and shear if triangle) based on board shape
        if self.board_type == "diamond":
            row = pos0 * math.cos(degree) + pos1 * math.sin(degree) 
            col = pos0 * math.sin(degree) - pos1 * math.cos(degree)
        else:
            # Rotate
            row = pos0 * math.cos(degree) - pos1 * math.sin(degree) 
            col = pos0 * math.sin(degree) + pos1 * math.cos(degree)
            # Shear / Skew
            row +=  (self.board_size - 1) * (col- (self.board_size - 1)) / (2 * (self.board_size - 1))
        
        return row,col

    def initialize(self, init_state):
        """ Initialize nodes and edges"""
        row_index = 0
        col_index = 0
        board_list = init_state["board"]
        self.board_size = int(math.sqrt(len(board_list)))
        #Generate nodes and add to graph
        for cell in board_list:
            # If triangle, only add valid nodes
            if  not (self.board_type  == "triangle" and col_index > row_index):
                self.G.add_node((row_index, col_index), **{'value': cell, 'node_color': node_colors[cell], 'node_size': NORMAL})
            
            #Index incrementation
            if col_index == self.board_size - 1:
                col_index = 0
                row_index += 1
            else:
                col_index += 1

        #Initialize edges in all valid directions for all valid nodes
        for row in range(0, self.board_size, 1):
            for col in range(0, self.board_size, 1):
                #If triangle, only edges between valid nodes
                if  self.board_type  == "triangle" and col > row:
                    continue
                #Add a tuple of (the node row,col , neighbor row,col) if the neighbor is within the board
                neighbors = [((row,col) , (row + d.vector[0], col + d.vector[1])) for d in self.directions if (row + d.vector[0] , col + d.vector[1]) in self.G.nodes()]
                self.G.add_edges_from(neighbors, **{'edge_color': "black"})
        # Show initial graph
        self.show_graph(len(self.game)<=1)


#test = Graphics()
#test.visualize_episode(test.game, "triangle", 1, 0.5)


