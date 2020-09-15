import scipy.sparse as sp
import numpy as np

class PreProcessing():
    """This class contains all methods required for preprocessing the
       vectors and matricies to allow a smooth transition between DwarfElephant
       and the external algorithm."""

    def read_csc_matrix(self,name):
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


    def read_vector(self,name):
        """Reads a vector from a text file exported by the MOOSE Framework

        Args: name = name of the vector to read in

        Returns: vector = Vector as numpy array.
        """

        vector = np.genfromtxt(name, skip_header=2, usecols=1)

        return vector

    def input_options(self,argument):
        """This method stores the possible text inputs for the function
        check_symmetry_and_modify().

        Args: argument = Number of the text input.

        Returns: Text input
        """

        inputs = {
            1: "The matrix is not symmetric. Therefore, it cannot be used in the following algorithm " +
               "without modification. Do ensure that the modfications are not changing the forward problem, " +
               "we guide you step by step through the preprocessing procedure. As a first check is your "+
               "bilinear form a) symmetric or b) not symmetric? \n" +
               "Please choose which of the above options is true.\n",
            2: "The option is currently not supported. \n",
            3: "Please choose a valid option. \n",
            4: "Does your forward problem has a) one or more Dirichlet boundary conditions or "+
               "b) no Dirichlet boundary condtitions? \n"+
               "Please choose which of the above options is true.\n",
            5: "Please check your forward problem again. There seems to be an error in the matrix generation.",
            6: "Is the Dirichlet boundary condition equal to a) zero or b) non-zero? "+
               "Please choose which of the above options is true.\n",
            7: "Please provide a the nodes related to the Dirichlet boundary condition."
            }
        return(inputs.get(argument))

    def check_symmetry(self,matrix, tol=1e-8):
        """We control if the matrix is symmetric.

        Args:
        matrix = The CSC matrix that has to be controlled and possibly modified.
        tol = The tolerance for the symmetry check.
        """
        if(np.max(np.abs(matrix-sp.csc_matrix.transpose(matrix)))<tol):
            print("The matrix is symmetric, no additional modifications are required.")
        else:
            print("The matrix is non-symmetric. Please, modify the matrix to a symmetric matrix.")

    def modify_matrix(self, matrix, nods_dirichlet=np.array([-1]), tol=1e-8, zero_rows=False, zero_cols=True):
        """In the case of a non-symmetric matrix the matrix is modified to a
        symmetric matrix if possible.

        Args:
        matrix = The CSC matrix that has to be controlled and possibly modified.
        nods_dirichlet = An array of the Nodes associated to the Dirichlet boundary
                         condition(s). The default value is -1.
        tol = The tolerance for the symmetry check.
        zero_rows = If true, we zero-out the rows associated to the Dirchlet boundary
                    condition(s). The default value is False.
        zero_cols = If true, we zero-out the columns associated to the Dirchlet boundary
                    condition(s). The default value is True.

        Returns: The symmetric matrix in CSC format.
        """

        print("We start to modify the matrix. Please wait.")
        matrix=matrix.tolil()
        if(zero_cols):
            matrix[:,nods_dirichlet]=0
        if(zero_rows):
            matrix[nods_dirichlet,:]=0
        matrix[nods_dirichlet,nods_dirichlet]=1
        print("The matrix is now modified.")
        matrix=matrix.tocsc()
        while True:
            if(np.max(np.abs(matrix-sp.csc_matrix.transpose(matrix)))<tol):
                print("The matrix is now symmetric. You can use the algorithm.")
                return(matrix)
            else:
                print("We are sorry something went wrong.")
                break

    def check_symmetry_and_modify(self,matrix, nods_dirichlet=np.array([-1]),tol=1e-8):
        """We control if the matrix is symmetric. In the case of a non-symmetric
        matrix the matrix is modified to a symmetric matrix if possible.

        Args:
        matrix = The CSC matrix that has to be controlled and possibly modified.
        nods_dirichlet = An array of the Nodes associated to the Dirichlet boundary
                         condition(s). The default value is -1.
        tol = The tolerance for the symmetry check.

        Returns: The symmetric matrix in CSC format.
        """
        if(np.max(np.abs(matrix-sp.csc_matrix.transpose(matrix)))<tol):
            print("The matrix is symmetric, no additional modifications are required.")
        else:
            option=input(self.input_options(1))
            while True:
                if(option=="a" or option=="a)"):
                    option=input(self.input_options(4))

                    if(option=="a" or option=="a)"):
                        option=input(self.input_options(6))
                        if(option=="a" or option=="a)"):
                            if(nods_dirichlet[0]==-1):
                                print(self.input_options(7))
                                break
                            else:
                                return self.modify_matrix(matrix,nods_dirichlet,tol)
                        elif(option=="b" or option=="b)"):
                            print(self.input_options(2))
                            break
                        else:
                            option=input("The option " + option + " is not a valid option." + input_options(3))
                    elif(option=="b" or option=="b)"):
                        print(self.input_options(5))
                        break
                    else:
                        option=input("The option " + option + " is not a valid option." + input_options(3))
                elif(option=="b" or option=="b)"):
                    print(self.input_options(2))
                    break
                else:
                    option=input("The option " + option + " is not a valid option." + input_options(3))
