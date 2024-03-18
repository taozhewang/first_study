# Collections

########################################################################
# Rational arithmetic
def add_rational(x, y):
    nx, dx = numer(x), denom(x)
    ny, dy = numer(y), denom(y)
    return rational(nx * dy + ny * dx, dx * dy)
def mul_rational(x, y):
    return rational(numer(x) * numer(y), denom(x) * denom(y))
def rationals_are_equal(x, y):
    return numer(x) * denom(y) == numer(y) * denom(x)
def print_rational(x):
    print(numer(x), "/", denom(x))
# Constructor and selectors
def rational(n, d): #   def rational(n, d):
    return [n, d]   #       def select(name):
def numer(x):       #           if name == 'n':
    return x[0]     #               return n
def denom(x):       #           elif name == 'd':
    return x[1]     #               return d
                    #       return select
                    #   def numer(x):
                    #       return x('n')
                    #   def denom(x):
                    #       return x('d')

################################################################
# Trees
def tree(label, branches = []):
    for branch in branches:
        assert is_tree(branch), 'branches must be lists'
    return [label] + list(branches)
def label(tree):
    return tree[0]
def branches(tree):
    return tree[1:]
def is_tree(tree):
    if type(tree) != list or len(tree) < 1:
        return False
    for branch in branches(tree):
        if not is_tree(branch):
            return False
    return True
def is_leaf(tree):
    return not branches(tree)
def count_leaves(t):
    if is_leaf(t):
        return 1
    else:
        return sum([count_leaves(b) for b in branches(t)]) 