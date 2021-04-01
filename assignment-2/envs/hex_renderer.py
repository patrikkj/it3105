
column_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def board2string(board):
    """
    From https://stackoverflow.com/questions/65396231/print-hex-game-board-contents-properly
    We ❤️ StackOverflow, temporary solution.
    """
    out = ["\n"]
    rows = len(board)
    cols = len(board[0])
    indent = 0
    headings = " "*5+(" "*3).join(column_names[:cols])
    out.append(headings)
    tops = " "*5+(" "*3).join("-"*cols)
    out.append(tops)
    roof = " "*4+"/ \\"+"_/ \\"*(cols-1)
    out.append(roof)
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


# def __str__(self):
#     return print_board(self.board)
#     # Lets make some fancy shear operations
#     lines = " " + np.array2string(self.board, 
#                             separator=' ', 
#                             prefix='', 
#                             suffix='').replace('[', '').replace(']', '')
#     return "\n".join(f"{' '*i}\\{line}\\" for i, line in enumerate(lines.split("\n")))