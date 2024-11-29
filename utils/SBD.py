import torch

# SBD (Similarity-Based Distance) calculation using NCC (Normalized Cross-Correlation)
def calcSBDncc(x, y, s):
    # Ensure the two input tensors have the same length and s is an integer
    assert x.size(0) == y.size(0)
    assert isinstance(s, int)
    
    length_ = x.size(0)
    
    # Calculate the sum of squares of each element in x and y (Euclidean norm)
    pow_x = torch.sum(x ** 2)
    pow_y = torch.sum(y ** 2)
    
    # Calculate the Euclidean distance of x and y
    dist_x = torch.sqrt(pow_x)
    dist_y = torch.sqrt(pow_y)
    dist_xy = dist_x * dist_y
    
    # Calculate cross-correlation between x and y at shift s
    ccs = torch.sum(x[s:length_] * y[0:length_ - s])  # Using slicing and element-wise multiplication
    
    # Calculate Normalized Cross-Correlation (NCC)
    ncc = ccs / dist_xy
    return ncc

def calcSBD(x, y, s=None):
    # Ensure the two input tensors have the same length
    assert x.size(0) == y.size(0)
    
    if s is None:
        # If s is not provided, calculate SBD for all shifts
        length_ = x.size(0)
        ncc_list = []
        for s in range(length_ - 1):
            ncc_list.append(calcSBDncc(x, y, s))
        
        # Find the maximum NCC value and calculate SBD
        ncc = max(ncc_list)
        t = ncc_list.index(ncc)  # Get the shift corresponding to the maximum NCC
        sbd = 1 - ncc  # Similarity-based distance
    else:
        # Calculate SBD for the given shift s
        ncc = calcSBDncc(x, y, s)
        sbd = 1 - ncc  # Similarity-based distance
        
    return sbd  # Return the calculated SBD
