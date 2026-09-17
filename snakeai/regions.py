"""Working out which parts of the board the snake can still get to.

Two questions get asked on every single step, for each of the three moves:

  how much room does this move lead into?   ->  free_straight / free_left / free_right
  could I still reach my own tail after it? ->  tail_straight / tail_left / tail_right

Both are answered by splitting the empty cells into connected regions, giving each one an id and
a size, then looking things up. Same region as the tail means the tail is reachable. Region size
capped at the snake's length is the room available.

If numba is installed the labelling runs compiled, which measured about 20x faster than the plain
Python version. If it is not, the plain version runs instead and gives identical answers. There
is a test for that.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError: # numba is optional, everything still works without it
    HAVE_NUMBA = False


def _label(blocked, entries, tail, budget, width, height, label, sizes, stack):
    """Split the empty cells into regions, then read off both answers.

    `blocked` is a flat width*height array, 1 where the body is. `entries` holds the flat index
    each move lands on, or -1 if that move is off the board or into the body. `label`, `sizes`
    and `stack` are scratch arrays passed in so they can be reused instead of reallocated.

    Returns (counts, tails), each three long, in the order straight, left, right.
    """
    n = width * height

    for i in range(n):
        label[i] = 0 # 0 means "not yet given a region"

    region = 0
    for start in range(n):
        if blocked[start] == 1 or label[start] != 0:
            continue # body, or already part of a region we found earlier

        # A cell nobody has reached yet, so it opens a new region. Spread out from it and claim
        # everything connected, counting as we go.
        region += 1
        label[start] = region
        stack[0] = start
        top = 1
        size = 0

        while top > 0:
            top -= 1
            here = stack[top]
            size += 1

            here_x = here % width
            here_y = here // width

            if here_x > 0: # west
                west = here - 1
                if blocked[west] == 0 and label[west] == 0:
                    label[west] = region
                    stack[top] = west
                    top += 1

            if here_x < width - 1: # east
                east = here + 1
                if blocked[east] == 0 and label[east] == 0:
                    label[east] = region
                    stack[top] = east
                    top += 1

            if here_y > 0: # north, y grows downward so this is minus a row
                north = here - width
                if blocked[north] == 0 and label[north] == 0:
                    label[north] = region
                    stack[top] = north
                    top += 1

            if here_y < height - 1: # south
                south = here + width
                if blocked[south] == 0 and label[south] == 0:
                    label[south] = region
                    stack[top] = south
                    top += 1

        sizes[region] = size

    # Every empty cell now knows its region, so both features are just lookups.
    counts = np.zeros(3, np.int32)
    tails = np.zeros(3, np.float32)
    tail_region = label[tail]

    for slot in range(3):
        if entries[slot] < 0:
            continue # that move is fatal, so it leads nowhere

        region_here = label[entries[slot]]
        counts[slot] = sizes[region_here] if sizes[region_here] < budget else budget
        if region_here == tail_region:
            tails[slot] = 1.0

    return counts, tails


# The plain Python version stays as the reference the tests check against.
label_regions_python = _label

if HAVE_NUMBA:
    label_regions = njit(cache=True)(_label)
else:
    label_regions = _label
