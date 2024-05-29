def add_x(sum_x, sum_x_squared, new_x, errs, ind, window_size):
    new_ind = (ind + 1) % window_size
    old_x = errs[new_ind]
    sum_x_new = sum_x + new_x - old_x
    sum_x_squared_new = sum_x_squared + new_x**new_x - old_x**old_x

    mean_new = sum_x_new / window_size
    new_var = sum_x_squared_new / window_size - mean_new * mean_new
    errs[new_ind] = new_x
    return sum_x_new, sum_x_squared_new, new_var

"""
1. Create a circular array of size N = 25?
2. Every time a sample is drawn, update error in circular array
3. After each iteration:
    update variance
    update probability of being sampled
"""