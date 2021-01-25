

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