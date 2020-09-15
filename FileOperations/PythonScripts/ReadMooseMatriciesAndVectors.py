import scipy.sparse as sp
import numpy as np

def read_csc_matrix(name):
    """Reads a sparse matrix from a text file exported by the MOOSE Framework

    Args: name = name of the matrix to read in

    Returns: csc_matrix = Sparse Matrix in CSC format.
    """

    # Read the matrix entries from file
    matrix_data=np.genfromtxt(name, skip_header=2)

    # Read the header with information about the number of rows and columns from file
    file = open(name, 'r')
    head = [next(file) for x in range(2)]
    m=int(head[0].split(":")[1])
    n=int(head[1].split(":")[1])

    # Generate the sparse matrix
    sparse_matrix=sp.csc_matrix((matrix_data[:,2],(matrix_data[:,0],matrix_data[:,1])), shape=(m,n))

    return sparse_matrix


def read_vector(name):
    """Reads a vector from a text file exported by the MOOSE Framework

    Args: name = name of the vector to read in

    Returns: vector = Vector as numpy array.
    """

    vector = np.genfromtxt(name, skip_header=2, usecols=1)

    return vector
