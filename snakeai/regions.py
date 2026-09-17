"""Working out which parts of the board the snake can still get to.

Three questions get asked on every single step, for each of the three moves:

  how much room does this move lead into?   ->  free_straight / free_left / free_right
  could I still reach my own tail after it? ->  tail_straight / tail_left / tail_right
  how open is it just around the corner?    ->  reach_straight / reach_left / reach_right

The first two are answered together. The empty cells get split into connected regions, each
with an id and a size. Same region as the tail means the tail is reachable. Region size,
capped at the snake's length, is the room available.

The third needs its own walk. Once two moves open into the same region the first two answers
give them the same number, so the third counts only what is a short walk away, which stays
different between them.

If numba is installed both run compiled, which measured about 20x faster than the plain
Python versions. If it is not, the plain versions run instead and give identical answers.
There is a test for that.
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

    blocked: flat width*height array, 1 where the body is.
    entries: the flat index each move lands on, or -1 if that move is off the board or into
        the body.
    tail: flat index of the tail cell, the one a move has to still reach.
    budget: region sizes are capped at this, so anything roomier than the snake reads the same.
    label, sizes, stack: scratch arrays passed in so they can be reused instead of reallocated.

    Returns (counts, tails), each three long, in the order straight, left, right.
    """
    cells = width * height # one flat index per board square

    for i in range(cells):
        label[i] = 0 # 0 means "not yet given a region"

    region = 0
    for start in range(cells):
        if blocked[start] == 1 or label[start] != 0:
            continue # body, or already part of a region we found earlier

        # A cell nobody has reached yet, so it opens a new region. Spread out from it and claim
        # everything connected, counting as we go.
        region += 1 # the id this new region gets
        label[start] = region # claim the cell we started from

        stack[0] = start # the only cell waiting to be spread out from
        top = 1 # cells waiting on the stack
        size = 0 # cells claimed for this region so far

        while top > 0:
            top -= 1 # take the cell off the top
            here = stack[top] # the cell we are spreading out from
            size += 1 # it belongs to this region

            here_x = here % width # column
            here_y = here // width # row

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

        sizes[region] = size # the region is fully claimed, so record how big it was

    # Every empty cell now knows its region, so both features are just lookups.
    counts = np.zeros(3, np.int32) # room each move leads into
    tails = np.zeros(3, np.float32) # 1.0 where the tail shares the region

    tail_region = label[tail] # the region the tail is sitting in

    for slot in range(3):
        if entries[slot] < 0:
            continue # that move is fatal, so it leads nowhere

        region_here = label[entries[slot]] # the region this move opens into
        counts[slot] = sizes[region_here] if sizes[region_here] < budget else budget # capped
        if region_here == tail_region:
            tails[slot] = 1.0 # the tail is in there with us

    return counts, tails


def _reach(blocked, entries, depth, width, height, seen, queue):
    """How many cells sit within `depth` moves of where each move lands.

    A different question from the one _label answers. Region size says whether a move leads
    anywhere survivable. Moves that open into the same region all get that same number, which
    on a mostly empty board is nearly all of them. Counting only what is close by stays
    different between moves, because it measures how hemmed in each one is right now.

    depth: how many steps out from the entry cell the count reaches.
    seen, queue: scratch arrays, reused between calls. `seen` is left all zero on the way out,
        so it never has to be cleared up front.

    Returns three counts, in the order straight, left, right.
    """
    counts = np.zeros(3, np.int32)

    for slot in range(3):
        if entries[slot] < 0:
            continue # that move is fatal, so it opens nothing up

        queue[0] = entries[slot] # the search starts from the cell this move lands on
        seen[entries[slot]] = 1 # and never goes back to it

        read = 0 # next cell to pop
        write = 1 # next free slot, which doubles as the count so far

        ring_left = 1 # cells left to pop at the depth we are on
        next_ring = 0 # cells found one step further out
        distance = 0 # rings stepped out from the entry cell so far

        while read < write:
            here = queue[read] # the cell we are spreading out from
            read += 1 # it has been popped
            ring_left -= 1 # one fewer waiting at this depth

            if distance < depth: # at the limit the queue still drains, it just stops growing
                here_x = here % width # column
                here_y = here // width # row

                if here_x > 0: # west
                    west = here - 1
                    if blocked[west] == 0 and seen[west] == 0:
                        seen[west] = 1
                        queue[write] = west
                        write += 1
                        next_ring += 1

                if here_x < width - 1: # east
                    east = here + 1
                    if blocked[east] == 0 and seen[east] == 0:
                        seen[east] = 1
                        queue[write] = east
                        write += 1
                        next_ring += 1

                if here_y > 0: # north, y grows downward so this is minus a row
                    north = here - width
                    if blocked[north] == 0 and seen[north] == 0:
                        seen[north] = 1
                        queue[write] = north
                        write += 1
                        next_ring += 1

                if here_y < height - 1: # south
                    south = here + width
                    if blocked[south] == 0 and seen[south] == 0:
                        seen[south] = 1
                        queue[write] = south
                        write += 1
                        next_ring += 1

            if ring_left == 0: # that was the last cell at this depth, so step outward
                distance += 1 # one further out from the entry cell
                ring_left = next_ring # what was found out there is the next ring
                next_ring = 0 # ready to collect the ring after that

        counts[slot] = write # every cell the queue ever held is what is within reach

        for i in range(write):
            seen[queue[i]] = 0 # hand the next move a clean array

    return counts


# ------------------------------------------------------------------- picking an implementation

# The plain Python versions stay as the reference the tests check against.
label_regions_python = _label # never compiled, whatever numba is doing
reach_counts_python = _reach # same, the slow but obvious answer

if HAVE_NUMBA:
    label_regions = njit(cache=True)(_label) # compiled on first call, cached on disk after
    reach_counts = njit(cache=True)(_reach) # same deal, the cache survives restarts
else:
    label_regions = _label # no numba, so the plain version is the only one there is
    reach_counts = _reach # likewise
