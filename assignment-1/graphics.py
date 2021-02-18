import math
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from utils import Direction

# Node sizes
SMALL = 40
NORMAL = 150
LARGE = 400

node_colors = {0 : 'black', 1: "blue"}
rotation = {
    "PegEnvironmentTriangle": -1/2 * math.pi, 
    "PegEnvironmentDiamond" : 3/4 * math.pi
}


class Graphics:
    def __init__(self, env, step_logs, delay=0.3):
        self.env = env
        self.step_logs = step_logs
        self.delay = delay
        self.game = []
        self.board_type = env.__class__.__name__
        self.G = nx.Graph()
        self.initialize()
    
    def initialize(self):
        """ Initialize nodes and edges"""
        #interactive mode pyplot for updating graph plot
        board = self.step_logs.iloc[0]["board"]
        
        #Generate nodes and add to graph
        for indices in self.env._valid_indices:
            value = board[indices]
            self.G.add_node(indices, **{'value': value, 'node_color': node_colors[value], 'node_size': NORMAL})

        #Initialize edges in all valid directions for all valid nodes
        for i, indices in enumerate(self.env._valid_indices):
            row, col = indices

            # Find all edges by trying to move in all directions
            neighbors = []
            for d in self.env.directions:
                if (row + d.vector[0], col + d.vector[1]) in self.G.nodes():
                    neighbors.append((indices, (row + d.vector[0], col + d.vector[1])))
            
            # Add all edges to graph
            self.G.add_edges_from(neighbors, **{'edge_color': "black"})

        # Show initial graph
        plt.ion()
        self.show_graph()        

    def visualize_episode(self, episode=-1):
        """Initializes and runs visualize_game on correct episode. Called by other classes."""
        # Filters step_logs to correct episode and converts to dict
        plt.ion()

        # Convert negative episode to backward counts
        if episode < 0:
            episode = self.step_logs["episode"].max() + (episode + 1)

        print("Visualizing ep:", episode)
        self.game = self.step_logs[self.step_logs["episode"] == episode]
        self.game = self.game.to_dict(orient="records")

        # Convert from numpy types
        for step in self.game:
            step["board"] =  step["board"].flatten().tolist()
            step["peg_start_position"] = tuple(step["peg_start_position"])
            step["peg_end_position"] = tuple(step["peg_end_position"])
            step["peg_move_direction"] = tuple(step["peg_move_direction"])
        
        #Loops through the step_states of the chosen episode and visualizes each step_state. """
        for step, step_state in enumerate(self.game[1:], start=1):
            self.visualize_step(step, step_state)

    def visualize_step(self, step, step_state):
        # Update value for cells that have changed, i.e. the ones involved in the peg move
        start_pos  = step_state["peg_start_position"]
        end_pos = step_state["peg_end_position"]
        vector = step_state["peg_move_direction"]
        middle_pos = (start_pos[0] + vector[0], start_pos[1] + vector[1])
    
        nx.set_node_attributes(self.G, {start_pos:  {'value': 0}})
        nx.set_node_attributes(self.G, {end_pos:  {'value': 1}})
        nx.set_node_attributes(self.G, {middle_pos:  {'value': 0}})

        #Update attributes of nodes the peg moved to and from
        if start_pos is not None:
            # Reset large and small nodes for last step's peg move
            nx.set_node_attributes(self.G, {self.game[step-1]["peg_start_position"] :  {'node_size': NORMAL}})
            nx.set_node_attributes(self.G, {self.game[step-1]["peg_end_position"] :  {'node_size': NORMAL}})

            # Reset edge color to black for all edges
            nx.set_edge_attributes(self.G, 'black', 'edge_color')

            # Update hole size and color of the three nodes involved in a move
            nx.set_node_attributes(self.G, {start_pos :  {'node_size': SMALL, 'node_color': node_colors[0]}})
            nx.set_node_attributes(self.G, {end_pos :  {'node_size': LARGE, 'node_color': node_colors[1]}})
            nx.set_node_attributes(self.G, {middle_pos :  {'node_color': node_colors[0]}})
            
            #Update edge color of edges along peg move path
            nx.set_edge_attributes(self.G,{(start_pos, middle_pos)  : {'edge_color': 'red'}})
            nx.set_edge_attributes(self.G,{(end_pos, middle_pos)  : {'edge_color': 'red'}})
        
        # final is a boolean that determines whether it's the last step. This is so that the last "frame" will hold for more than self.delay seconds
        final = step == (len(self.game) - 1)
        self.show_graph(final=final)

    def show_graph(self, final=False):
        """Plots the graph with nodes and edges and pauses for self.delay time."""
        color_list = []
        size_list = []
        first_pos_list = {}
        pos_dict = {}
        degree = rotation[self.board_type]

        # Collect lists of node colors and node_sizes that will be arguments in the nx.draw_networkx function
        first_pos_list = {k: k for k, v in self.G.nodes(data=True)}
        for key, attributes in self.G.nodes(data=True):
            color_list.append(attributes["node_color"])
            size_list.append(attributes["node_size"])
        
        # Edges         
        edges = self.G.edges()
        edge_colors = [self.G[u][v]["edge_color"] for u,v in edges]

        # Dict of positions of each node with node index as key. Rotated and sheared (if triangle) coordinates that will be for the pos argument in nx.draw_networkx function
        for pos in first_pos_list:
            row, col = self.rotate(pos, degree)
            pos_dict[pos] = (row,col)

        fig = plt.figure("Peg Solitaire")
        nx.draw_networkx(self.G, pos=pos_dict, with_labels= False, node_color = color_list, node_size = size_list, edge_color = edge_colors)
        if final:
            plt.pause(1e4)    
            plt.ioff()        
        else:
            plt.pause(self.delay)
        plt.clf()

    def rotate(self, pos, degree):
        """Rotates and shears the position coordinates of the nodes in the graph depending on the board type."""
        pos0, pos1 = pos

        # Rotation (and shear if triangle) based on board shape
        if self.board_type == "PegEnvironmentDiamond":
            row = pos0 * math.cos(degree) + pos1 * math.sin(degree) 
            col = pos0 * math.sin(degree) - pos1 * math.cos(degree)
        elif self.board_type == "PegEnvironmentTriangle":
            # Rotate
            row = pos0 * math.cos(degree) - pos1 * math.sin(degree) 
            col = pos0 * math.sin(degree) + pos1 * math.cos(degree)

            # Shear / Skew
            n = self.env.board_size
            row +=  (n - 1) * (col- (n - 1)) / (2 * (n - 1))
        return row, col
