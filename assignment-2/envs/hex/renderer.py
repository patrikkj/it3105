
from functools import lru_cache
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, RegularPolygon


class HexRenderer:
    # Static variables
    THETA = np.radians(30)
    edge_color = "#606060"
    hex_colors = {
        0.: "#bbb",
        1.: "#F8A857",
        2.: "#57A8F8"
    }
    tri_colors = {
        1: "#FAC48E",
        2: "#85BFF9"
    }

    @staticmethod
    @lru_cache(maxsize=1)
    def _transform_matrix():
        """Applies an affine transformation which maps to the hex coordinate system."""
        # TODO: Rewrite transformations to using 'matplotlib.transforms.Affine2D'
        theta = HexRenderer.THETA
        cos, sin = np.cos(theta), np.sin(theta)

        # Construct affine transformations
        skew_matrix = np.array([[1, 0], [0, np.sqrt(3)/2]])     # Adjust 'y' coordinates for compact grid layout
        shear_matrix = np.array([[1, 0], [theta, 1]])
        rotation_matrix = np.array(((cos, -sin), (sin, cos)))
        return skew_matrix @ shear_matrix @ rotation_matrix

    @staticmethod
    def render(board, block=True, pause=0.1, close=True, title=None, callable_=None):
        # Create figure
        if plt.get_fignums():
            plt.close()
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')

        # Create ramdonly filled board to see colors
        #board = np.random.randint(0, 3, board.shape)
        n = board.shape[0]
        low, high = 0, n - 1
        label_offset = 1
        tri_offset = 1.5

        # Add hexagons
        old_coords = np.indices(board.shape).astype(float).reshape(2, -1).T
        coords = old_coords @ HexRenderer._transform_matrix()

        for action, coord, player in zip(range(n**2), coords, board.flat):
            hexagon = RegularPolygon(coord, numVertices=6, radius=np.sqrt(1/3), orientation=HexRenderer.THETA, 
                facecolor=HexRenderer.hex_colors[player], edgecolor=HexRenderer.edge_color, zorder=1, picker=True)
            hexagon.action = action
            ax.add_patch(hexagon)
        
        # Add labels for cell references (A1, B3, ...)
        label_low, label_high = low - label_offset, high + label_offset
        top_coords = np.vstack([np.full(n, fill_value=label_low), np.arange(n)]).T
        bottom_coords = np.vstack([np.full(n, fill_value=label_high), np.arange(n)]).T
        alpha_coords = np.vstack([top_coords, bottom_coords]) @ HexRenderer._transform_matrix()
        alpha_labels = np.tile(np.array(list(ascii_uppercase[:n])), 2)

        left_coords = np.vstack([np.arange(n), np.full(n, fill_value=label_low)]).T
        right_coords = np.vstack([np.arange(n), np.full(n, fill_value=label_high)]).T
        numeric_coords = np.vstack([left_coords, right_coords]) @ HexRenderer._transform_matrix()
        numeric_labels = np.tile(np.array(list(map(str, range(1, n+1)))), 2)

        for label, coords in zip([*alpha_labels, *numeric_labels], [*alpha_coords, *numeric_coords]):
            ax.text(*coords, label, fontsize=7, fontweight='bold', ha="center", va="center")

        # Add triangles in the background
        tri_low, tri_high = low - tri_offset, high + tri_offset
        tri_coords = np.array([
            (tri_high, tri_high), (tri_low, tri_low), 
            (tri_low, tri_high), (tri_high, tri_low), 
            ((tri_low + tri_high)/2, (tri_low + tri_high)/2)
        ])
        tri_coords = tri_coords @ HexRenderer._transform_matrix()
        t, b, l, r, c = tri_coords
        triangles = [((t, l, c), 1), ((r, b, c), 1), ((l, b, c), 2), ((r, t, c), 2)]

        for tri_coords, player in triangles:
            triangle = Polygon(tri_coords, facecolor=HexRenderer.tri_colors[player], edgecolor=HexRenderer.edge_color, zorder=0)
            ax.add_patch(triangle)
    
        # Display figure
        if title:
            plt.title(title)
        if callable_:
            fig.canvas.mpl_connect('pick_event', callable_)
        plt.autoscale(enable=True)
        plt.axis('off')
        plt.show(block=block)
        if pause:
            plt.pause(pause)
        if close:
            plt.close()

    @staticmethod
    def board2string(board):
        """
        From https://stackoverflow.com/questions/65396231/print-hex-game-board-contents-properly
        We ❤️ StackOverflow, temporary solution.
        """
        out = ["\n"]
        rows = len(board)
        cols = len(board[0])
        indent = 0
        headings = " "*5+(" "*3).join(ascii_uppercase[:cols])
        out.append(headings)
        out.append(" "*5+(" "*3).join("-"*cols))    # tops
        out.append(" "*4+"/ \\"+"_/ \\"*(cols-1))   # roof
        BLUE = '\x1b[0;0;43m \x1b[0m'
        RED = '\x1b[0;0;41m \x1b[0m'
        color_mapping = lambda i : (' ', BLUE, RED)[i]
        for r in range(rows):
            row_mid = " "*indent
            row_mid += " {} | ".format(r+1)
            row_mid += " | ".join(map(color_mapping,board[r]))
            row_mid += " | {} ".format(r+1)
            out.append(row_mid)
            row_bottom = " "*indent
            row_bottom += " "*3+" \\_/"*cols
            if r<rows-1:
                row_bottom += " \\"
            out.append(row_bottom)
            indent += 2
        headings = " "*(indent-2)+headings
        out.append(headings)
        return "\n".join(out)

    @staticmethod
    def board2asciistring(board):
       # Lets make some fancy shear operations
       lines = " " + np.array2string(board, 
                               separator=' ', 
                               prefix='', 
                               suffix='').replace('[', '').replace(']', '')
       return "\n".join(f"{' '*i}\\{line}\\" for i, line in enumerate(lines.split("\n")))
