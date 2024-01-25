import math

def n_codebooks_to_bandwidth_id(n_codebooks):
    return max(0, math.ceil(math.log2(n_codebooks))-1)