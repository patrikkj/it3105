

import numpy as np
v = np.random.random(2**25)

u = np.random.random(2**25)
import time

t0 = time.perf_counter()

for _ in range(20):
    v = v + 2.3*0.2* u
    v = v + 2.17*u
    print(v)

t1 = time.perf_counter()
print(t1-t0)


        # return "\n".join("".join(map(str, row)) for row in self.board.astype(int))





    #def _generate_moves(self):      
    #    conv = ndimage.convolve(self.board + self._mask, kernel, mode="constant", cval=1)
    #    conv = np.bitwise_xor(conv, edge_mask) #Flipper bitene som indikerer ytterkanten (altså to steg fra current cell)
    #    if self.board_type == "triangle":
    #        conv = conv * np.tri(self.board_size, dtype=np.int64) #*np.tri(board_size) sørger for at alle ugyldige states (oppe høyre) blir lik 0. Hensikt: sørge for at de ikke kan "leve" og flytte inn på lovlig felt.
    #    for i, direction in enumerate(self.directions):
    #        indices = (np.bitwise_and(conv, direction.kernel) == direction.kernel).nonzero() #sjekker om andet direction med state gir direction. Da vil direction-flyttet være godkjent.
    #        yield from ((x, y, i) for x, y in zip(*indices))  # Slår sammen til en liste med tupler i stedet for to lister av arrays.




    #def _generate_moves(self):
    #    temp_board = self.board + self._mask
    #    for index, v in np.ndenumerate(self.board):
    #        if v == 0:
    #            continue
    #        x, y = index
    #        for i, direction in enumerate(self.directions):
    #            dx, dy = direction.vector
    #            try:
    #                if temp_board[x+dx, y+dy] == 1 and \
    #                    temp_board[x+2*dx, y+2*dy] == 0:
    #                    yield (x, y, i)
    #            except:
    #                pass