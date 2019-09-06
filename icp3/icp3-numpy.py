
import numpy as np


# create random vector of size 15 having only Integers in the range 1-20.

rand1 = np.random.randint(20, size=15)
print(rand1)

# reshape the array to 3 by 5
rand = rand1.reshape((3,5))
print(rand)


# replace the max in each row by 0
rem_maxes = rand.max(axis=1).reshape(-1, 1)
print(rem_maxes)
# np.where(rand == rem_maxes, 1, 0)
rand[:] = np.where(rand == rem_maxes, 0, rand)
print(rand)

