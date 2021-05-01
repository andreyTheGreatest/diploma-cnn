import numpy as np

 
def convolution(image, kernel, bias, s=1):
    '''
    Main CNN function

    The output matrix values show high connectivity
    between input matrix and kernel matrix 
    (describes a feature to be found).
    If a value is high, hence the presence of given
    feature in input matrix is probable. It is though
    unlikely to be if the value is close to 0 or negative.

    '''
    (n_f, n_c_f, f, _) = kernel.shape
    n_c, in_dim, _ = image.shape

    out_dim = int((in_dim - f) / s) + 1
    
    assert n_c == n_c_f

    out = np.zeros((n_f, out_dim, out_dim))

    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            while curr_x + f <= in_dim:
                out[curr_f, out_y, out_x] = np.sum(kernel[curr_f] * image[:, curr_y : curr_y + f, curr_x : curr_x + f]) + bias[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

    return out

def maxpool(image, f=2, s=2):
    '''
    Common optimization function

    Reduces the shape of input matrix by 
    recognising its relevent neurons simply
    choosing greater values whilst 
    neglecting lower ones.

    '''
    n_c, h_prev, w_prev = image.shape
    h = int((h_prev - f) / s) + 1
    w = int((w_prev - f) / s) + 1

    max_pooled = np.zeros((n_c, h, w))

    for i in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:
                max_pooled[i, out_y, out_x] = np.max(image[i, curr_y : curr_y + f, curr_x : curr_x + f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    
    return max_pooled

def softmax(raw_predictions):
    '''
    Softmax function (TODO)

    Maps all the final dense layer outputs 
    to a vector whose elements sum up to one

    '''
    out = np.exp(raw_predictions)
    return out/np.sum(out)

def categoricalCrossEntropy(probs, label):
    '''
    calculate the categorical cross-entropy loss of the predictions
    '''
    return -np.sum(label * np.log(probs))