import numpy as np
import itertools

# Given map
grid = np.array([
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1, 1]
])

# List of possible actions defined in terms of changes in
# the coordinates (y, x)
forward = [
    (-1,  0),   # Up
    ( 0, -1),   # Left
    ( 1,  0),   # Down
    ( 0,  1),   # Right
]

# Three actions are defined:
# - right turn & move forward
# - straight forward
# - left turn & move forward
# Note that each action transforms the orientation along the
# forward array defined above.
action = [-1, 0, 1]
action_name = ['R', '#', 'L']

init = (4, 3, 0)    # Representing (y, x, o), where
                    # o denotes the orientation as follows:
                    # 0: up
                    # 1: left
                    # 2: down
                    # 3: right
                    # Note that this order corresponds to forward above.
goal = (2, 0)
cost = (2, 1, 20)   # Cost for each action (right, straight, left)

# EXAMPLE OUTPUT:
# calling optimum_policy_2D with the given parameters should return
# [[' ', ' ', ' ', 'R', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', '#'],
#  ['*', '#', '#', '#', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', ' '],
#  [' ', ' ', ' ', '#', ' ', ' ']]

def optimum_policy_2D(grid, init, goal, cost):
    # Initialize the value function with (infeasibly) high costs.
    value = np.full((4, ) + grid.shape, 999, dtype=np.int32)
    # Initialize the policy function with negative (unused) values.
    policy = np.full((4,) + grid.shape, -1, dtype=np.int32)
    # Final path policy will be in 2D, instead of 3D.
    policy2D = np.full(grid.shape, ' ')

    # Apply dynamic programming with the flag change.
    change = True
    while change:
        change = False
        # This will provide a useful iterator for the state space.
        p = itertools.product(
            range(grid.shape[0]),
            range(grid.shape[1]),
            range(len(forward))
        )
        # Compute the value function for each state and
        # update policy function accordingly.
        for y, x, t in p:
            # Mark the final state with a special value that we will
            # use in generating the final path policy.
            if (y, x) == goal and value[(t, y, x)] > 0:
                # TODO: implement code.
                change = True
                value[(t, y, x)] = 0
                policy[(t, y, x)] = -999
            # Try to use simple arithmetic to capture state transitions.
            elif grid[(y, x)] == 0:
                # TODO: implement code.
                for i in range(len(action)):
                    temp_t = (t + action[i]) % 4
                    temp_y = y + forward[temp_t][0]
                    temp_x = x + forward[temp_t][1]
                    
                    if 0 <= temp_x < grid.shape[1] and 0 <= temp_y < grid.shape[0] and grid[(temp_y, temp_x)] == 0:
                        temp_cost = value[(temp_t, temp_y, temp_x)] + cost[i]

                        if temp_cost < value[(t, y, x)]:
                            value[(t, y, x)] = temp_cost
                            policy[(t, y, x)] = action[i]
                            change = True

                
    # Now navigate through the policy table to generate a
    # sequence of actions to take to follow the optimal path.
    # TODO: implement code.
    y, x, t = init
    policy2D[(y, x)] = action_name[policy[(t, y, x)] + 1]

    while policy[(t, y, x)] != -999:
        if policy[(t, y, x)] == action[0]:
            t = (t - 1) % 4
        elif policy[(t, y, x)] == action[2]:
            t  = (t + 1) % 4
        
        y += forward[t][0]
        x += forward[t][1]

        if policy[(t, y, x)] == -999:
            policy2D[(y, x)] = '*'
        else:
            policy2D[(y, x)] = action_name[policy[(t, y, x)] + 1]


    # Return the optimum policy generated above.
    return policy2D

print(optimum_policy_2D(grid, init, goal, cost))
