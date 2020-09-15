import xdrlib
import numpy as np
import scipy.linalg as la

class OnlineStage:
    def read_offline_data(self,path_offline_data, q_a, q_f, q_l, n_outputs):
        """This method is responsible for reading all offline data that are necessary for
           calculating the output of interest without the output error bounds.

           Args:
                path_offline_data = path to the offline data folder
                q_a = number of stiffness matrices (A)
                q_f = number of load vectors (f)
                q_l = number of attached theta objects to each output vector
                n_outputs = number of output vectors (l)

            Returns:
                n_bfs = number of basis functions
                RB_Aq = reduced stiffness matrices
                RB_Fq = reduced load vectors
                RB_Oq = reduced load vectors

            """
        # number of basis functions
        with open(path_offline_data + '/n_bfs.xdr', 'rb') as f_reader:
            f = f_reader.read()
            n_bfs = xdrlib.Unpacker(f).unpack_int()

        # RB_Aq
        RB_Aq = np.empty([n_bfs, n_bfs, q_a])
        for i in range(q_a):
            f = open('{}/RB_A_{}.xdr'.format(path_offline_data, str(i).zfill(3)), 'rb').read()
            u = xdrlib.Unpacker(f)
            orig_array = u.unpack_farray(n_bfs*n_bfs,u.unpack_double)
            temp = np.reshape(orig_array,[n_bfs,n_bfs])
            RB_Aq[:,:,i] = temp

        # RB_Fq
        RB_Fq = np.empty([n_bfs, q_f])
        for i in range(q_f):
            f = open('{}/RB_F_{}.xdr'.format(path_offline_data, str(i).zfill(3)), 'rb').read()
            u = xdrlib.Unpacker(f)
            RB_Fq[:,i] = u.unpack_farray(n_bfs,u.unpack_double)

        # RB_Oq
        RB_Oq = np.empty([n_bfs, n_outputs, q_l])
        for i in range(n_outputs):
            for j in range(q_l):
                f = open("{}/output_{}_{}.xdr".format(path_offline_data,
                            str(i).zfill(3), str(j).zfill(3)), 'rb').read()
                u = xdrlib.Unpacker(f)
                RB_Oq[:,i,j] = u.unpack_farray(n_bfs,u.unpack_double)


        return (n_bfs, RB_Aq, RB_Fq, RB_Oq)

    def read_error_offline_data(path_offline_data, q_a, q_f, q_l, n_outputs,
                                online_N):
        """This method is responsible for reading all offline data that are
           additionally necessary for calculating the output error bounds.

           Args:
                path_offline_data = path to the offline data folder
                q_a = number of stiffness matrices (A)
                q_f = number of load vectors (f)
                q_l = number of attached theta objects to each output vector
                n_outputs = number of output vectors (l)
                online_N = the number of basis functions that should be considered

            Returns:
                Fq_representor_innerprods = the innerproducts from Fq-Fq
                Fq_Aq_representor_innerprods = the innerproducts from Fq-Aq
                Aq_Aq_representor_innerprods = the innerproducts from Aq-Aq
                output_dual_innerprods = the innerproducts from output-output

            """
        f = open(path_offline_data + '/Fq_innerprods.xdr').read()
        u = xdrlib.Unpacker(f)
        Fq_representor_innerprods = np.zeros([(q_f*(q_f+1)/2),1])
        Fq_representor_innerprods = np.reshape(u.unpack_farray((q_f*(q_f+1)/2),
                                               u.unpack_double),
                                               [(q_f*(q_f+1)/2),1])

        Fq_Aq_representor_innerprods = np.zeros([q_f,q_a,online_N])
        f = open(path_offline_data + '/Fq_Aq_innerprods.xdr').read()
        u = xdrlib.Unpacker(f)
        Fq_Aq_representor_innerprods = np.reshape(
                            u.unpack_farray(q_f*q_a*online_N,u.unpack_double),
                            [q_f,q_a,online_N])

        Aq_Aq_representor_innerprods = np.zeros([(q_a*(q_a+1)/2),online_N,
                                                online_N])
        f = open(path_offline_data + '/Aq_Aq_innerprods.xdr').read()
        u = xdrlib.Unpacker(f)
        Aq_Aq_representor_innerprods = np.reshape(
                            u.unpack_farray((q_a*(q_a+1)/2)*online_N*online_N,
                                            u.unpack_double),
                                            [(q_a*(q_a+1)/2),online_N, online_N])

        output_dual_innerprods = np.zeros([n_outputs, (q_l*(q_l+1)/2)])
        for i in range(n_outputs):
            f = open(path_offline_data + '/output_00' + str(i) +
                     '_dual_innerprods.xdr').read()
            u = xdrlib.Unpacker(f)
            output_dual_innerprods[i,:] = u.unpack_farray((q_l*(q_l+1)/2),
                                                            u.unpack_double)

        return (Fq_representor_innerprods, Fq_Aq_representor_innerprods,
                Aq_Aq_representor_innerprods, output_dual_innerprods)

    def read_transient_offline_data(path_transient_offline_data, q_m, n_bfs,
                                    parameter_dependent_IC = False, q_ic = 0):
        """This method is responsible for reading all offline data that are
           additionally necessary for the transient output of interest without the
           output error bounds.

           Args:
                path_offline_data = path to the transient offline data folder
                q_m = number of mass matrices (M)
                n_bfs = number of basis functions
                parameter_dependent_IC = determines whether the initial conditions
                                         are parameter dependent or note
                q_ic = number of intial conditions (IC)

            Returns:
                RB_Mq = reduced mass matrices
                initial_conditions = initial conditions
                RB_L2_matrix = reduced L2 matrix (only returned if
                               parameter_dependent_IC = True)

            """
        # RB_Mq
        RB_Mq = np.ndarray([n_bfs, n_bfs, q_m])
        if(parameter_dependent_IC==True):
            initial_conditions = np.ndarray([n_bfs, q_ic])
            RB_L2_matrix = np.ndarray([n_bfs, n_bfs])

        for i in range(q_m):
            f = open(path_transient_offline_data + '/RB_M_00' + str(i) +
                     '.xdr').read()
            u = xdrlib.Unpacker(f)
            RB_Mq[:,:,i] = np.reshape(u.unpack_farray(n_bfs*n_bfs,u.unpack_double),
                                      [n_bfs,n_bfs])

        # intial conditions
        # currently it is only supported if online_N = n_bfs
        if(parameter_dependent_IC == False):
            # initial_conditions = np.ndarray([40,1])
            f = open(path_transient_offline_data + '/initial_conditions.xdr').read()
            u = xdrlib.Unpacker(f)
            position = np.sum(np.arange(1,n_bfs,1))

            initial_conditions = u.unpack_farray(position+n_bfs,u.unpack_double)
            initial_conditions = initial_conditions[position:]

            return (RB_Mq, initial_conditions)

        else:
            for i in range(q_ic):
                f = open(path_transient_offline_data + '/RB_IC_00' + str(i) +
                         '.xdr').read()
                u = xdrlib.Unpacker(f)
                initial_conditions[:,i] = u.unpack_farray(n_bfs,u.unpack_double)

            f = open(path_transient_offline_data + '/RB_L2_matrix.xdr').read()
            u = xdrlib.Unpacker(f)
            RB_L2_matrix = np.reshape(u.unpack_farray(n_bfs*n_bfs, u.unpack_double),
                                           [n_bfs,n_bfs])

            return (RB_Mq, initial_conditions, RB_L2_matrix)

    def read_transient_error_offline_data(path_transient_offline_data, n_bfs, q_a,
                                          q_m, q_f):
        """This method is responsible for reading all offline data that are
           additionally necessary for the transient output of interest with the
           output error bounds.

           Args:
                path_offline_data = path to the transient offline data folder
                n_bfs = number of basis functions
                q_a = number of stiffness matrices (A)
                q_m = number of mass matrices (M)
                q_f = number of load vectors (f)

            Returns:
                initial_L2_error = initial L2 error
                Fq_Mq_representor_innerprods = the innerproducts from Fq-Mq
                Aq_Mq_representor_innerprods = the innerproducts from Aq-Mq
                Mq_Mq_representor_innerprods = the innerproducts from Mq-Mq

            """
        # intial L2 error
        # currently it is only supported if online_N = n_bfs
        f = open(path_transient_offline_data + '/initial_L2_error.xdr').read()
        u = xdrlib.Unpacker(f)
        initial_L2_error = u.unpack_farray(n_bfs,u.unpack_double)
        initial_L2_error = initial_L2_error[-1]

        f = open(path_transient_offline_data + '/Fq_Mq_terms.xdr').read()
        u = xdrlib.Unpacker(f)
        Fq_Mq_representor_innerprods = np.reshape(
                            u.unpack_farray(q_f*q_m*n_bfs,u.unpack_double),
                                            [q_f,q_m,n_bfs])

        f = open(path_transient_offline_data + '/Aq_Mq_terms.xdr').read()
        u = xdrlib.Unpacker(f)
        Aq_Mq_representor_innerprods = np.reshape(
                                    u.unpack_farray(q_a*q_m*n_bfs*n_bfs,
                                                    u.unpack_double),
                                                    [q_a, q_m ,n_bfs, n_bfs])

        f = open(path_transient_offline_data + '/Mq_Mq_terms.xdr').read()
        u = xdrlib.Unpacker(f)
        Mq_Mq_representor_innerprods = np.reshape(
                            u.unpack_farray((q_m*(q_m+1)/2)*n_bfs*n_bfs,
                                            u.unpack_double),
                                            [(q_m*(q_m+1)/2),n_bfs, n_bfs])

        return (initial_L2_error, Fq_Mq_representor_innerprods,
                Aq_Mq_representor_innerprods, Mq_Mq_representor_innerprods)


    def read_basis_functions(self,path_basis_functions, online_N):
        """This method reads the basis functions exported from DwarfElephant.
            Args:
                path_basis_functions = path to the basis functions
                online_N = the number of basis functions that should be considered
            Returns:
                basis_functions = The basis functions as numpy array
        """

        basis_functions = np.genfromtxt(path_basis_functions + str(0), delimiter='\t',
                                            skip_header = 2, usecols=1)
        for i in range(1,online_N):
            basis_function = np.genfromtxt(path_basis_functions + str(i), delimiter='\t',
                                            skip_header = 2, usecols=1)
            basis_functions = np.vstack((basis_functions, basis_function))
        return basis_functions

    def rb_solve_without_error_bound(self,online_mu_parameters, q_a, q_f, q_l, n_outputs, online_N,
                                 RB_Aq, RB_Fq, RB_Oq):
        """This method is responsible performing the rb solve without the output error bounds.

           Args:
               online_mu_parameters = online parameters
               q_a = number of stiffness matrices (A)
               q_f = number of load vectors (f)
               q_l = number of attached theta objects to each output vector
               n_outputs = number of output vectors (l)
               online_N = the number of basis functions that should be considered
               RB_Aq = reduced stiffness matrices
               RB_Fq = reduced load vectors
               RB_Oq = reduced load vectors

            Returns:
                RB_outputs = the output of interest

            """

        # assemble the RB system
        RB_system_matrix = np.sum(RB_Aq*self.theta_A(online_mu_parameters), axis = 2)

        # assemble the RB rhs
        RB_rhs = np.sum(RB_Fq*self.theta_F(online_mu_parameters), axis = 1)

        RB_solution = np.reshape(la.lu_solve(la.lu_factor(RB_system_matrix), RB_rhs),[online_N,1])

        # evaluate the RB outputs
        RB_outputs = np.sum(RB_solution*(RB_Oq[:,:,0]*self.theta_O(online_mu_parameters)), axis=0)

        return  (RB_outputs)

    def rb_state_solve_without_error_bound(self,online_mu_parameters, q_a, q_f, online_N,
                                 RB_Aq, RB_Fq):
        """This method is responsible performing the rb solve without the output error bounds.

           Args:
               online_mu_parameters = online parameters
               q_a = number of stiffness matrices (A)
               q_f = number of load vectors (f)
               q_l = number of attached theta objects to each output vector
               n_outputs = number of output vectors (l)
               online_N = the number of basis functions that should be considered
               RB_Aq = reduced stiffness matrices
               RB_Fq = reduced load vectors
               RB_Oq = reduced load vectors

            Returns:
                RB_solution = the state with the dimension of the RB space

            """

        # assemble the RB system
        RB_system_matrix = np.sum(RB_Aq*self.theta_A(online_mu_parameters), axis = 2)

        # assemble the RB rhs
        RB_rhs = np.sum(RB_Fq*self.theta_F(online_mu_parameters), axis = 1)

        RB_solution = np.reshape(la.lu_solve(la.lu_factor(RB_system_matrix), RB_rhs),[online_N,1])

        return  (RB_solution)


def compute_residual_dual_norm(Fq_inner, Fq_Aq_inner, Aq_Aq_inner, q_a, q_f,
                               online_mu_parameters, online_N, RB_solution):
    """This method is responsible for computing the residual dual norm.

       Args:
            Fq_inner = the innerproducts from Fq-Fq
            Fq_Aq_inner = the innerproducts from Fq-Aq
            Aq_Aq_inner = the innerproducts from Aq-Aq
            q_a = number of stiffness matrices (A)
            q_f = number of load vectors (f)
            online_mu_parameters = online parameters
            online_N = the number of basis functions that should be considered
            RB_solution = reduced solution vector

        Returns:
            residual_norm_sq = the square root of the residual dual norm

        """
    residual_norm_sq = 0.
    q=0
    for q_f1 in range(q_f):
        for q_f2 in range(q_f1, q_f):
            if(q_f1==q_f2):
                delta = 1.
            else:
                delta = 2.

            residual_norm_sq += delta*np.real(theta_F(online_mu_parameters)[q_f1]*
                                              np.conj(theta_F(online_mu_parameters[q_f2]))*
                                                      Fq_inner[q])
            q +=1

    for q_f1 in range(q_f):
        for q_a1 in range(q_a):
            for i in range(online_N):
                delta = 2.

                residual_norm_sq += delta*np.real(theta_F(online_mu_parameters)[q_f1]*
                                           np.conj(theta_A(online_mu_parameters)[q_a1])*
                                           np.conj(RB_solution[i]*Fq_Aq_inner[q_f1][q_a1][i]))

    q=0
    for q_a1 in range(q_a):
        for q_a2 in range(q_a1, q_a):
            if(q_a1==q_a2):
                delta = 1.
            else:
                delta = 2.

            for i in range(online_N):
                for j in range(online_N):
                    residual_norm_sq += delta*np.real(np.conj(
                                        theta_A(online_mu_parameters)[q_a1])*
                                        theta_A(online_mu_parameters[q_a2])*
                                        np.conj(RB_solution[i]) * RB_solution[j] *
                                                  Aq_Aq_inner[q][i][j])
            q +=1

    if(np.real(residual_norm_sq)<0.):
        residual_norm_sq = abs(residual_norm_sq)

    return np.sqrt(residual_norm_sq)

def cache_online_residual_terms(online_N, q_a, q_m, q_f, Fq_inner, Fq_Aq_inner, Aq_Aq_inner,
                                Fq_Mq_inner, Aq_Mq_inner, Mq_Mq_inner, online_mu_parameters):

    """This method is responsible for caching the online residual terms.

       Args:
            online_N = the number of basis functions that should be considered
            q_a = number of stiffness matrices (A)
            q_m = number of mass matrices (M)
            q_f = number of load vectors (f)
            Fq_inner = the innerproducts from Fq-Fq
            Fq_Aq_inner = the innerproducts from Fq-Aq
            Aq_Aq_inner = the innerproducts from Aq-Aq
            Fq_Mq_inner = the innerproducts from Fq-Mq
            Aq_Mq_inner = the innerproducts from Aq-Mq
            Mq_Mq_inner = the innerproducts from Mq-Mq
            online_mu_parameters = online parameters

        Returns:
            cached_Fq_term = Fq contribution to residual dual norm
            cached_Fq_Aq_vector = Fq-Aq contribution to residual dual norm
            cached_Aq_Aq_matrix = Aq-Aq contribution to residual dual norm
            cached_Fq_Mq_vector = Fq-Mq contribution to residual dual norm
            cached_Aq_Mq_matrix = Aq-Mq contribution to residual dual norm
            cached_Mq_Mq_matrix = Mq-Mq contribution to residual dual norm

    """
    cached_Fq_term = 0
    q=0
    for q_f1 in range(q_f):
        for q_f2 in range(q_f1, q_f):
            if(q_f1==q_f2):
                delta = 1.
            else:
                delta = 2.

            cached_Fq_term += delta*theta_F(online_mu_parameters)[q_f1] \
            *theta_F(online_mu_parameters)[q_f2]*Fq_inner[q]
            q +=1

    cached_Fq_Aq_vector = np.zeros([online_N,1])
    for q_f1 in range(q_f):
        for q_a1 in range(q_a):
            for i in range(online_N):
                delta = 2.

                cached_Fq_Aq_vector[i] += delta*theta_F(online_mu_parameters)[q_f1] \
                *theta_A(online_mu_parameters)[q_a1] * Fq_Aq_inner[q_f1][q_a1][i]


    cached_Aq_Aq_matrix = np.zeros([online_N, online_N])
    q=0
    for q_a1 in range(q_a):
        for q_a2 in range(q_a1, q_a):
            if(q_a1==q_a2):
                delta = 1.
            else:
                delta = 2.

            for i in range(online_N):
                for j in range(online_N):
                    cached_Aq_Aq_matrix[i,j] += delta*theta_A(online_mu_parameters)[q_a1] \
                    *theta_A(online_mu_parameters)[q_a2]*Aq_Aq_inner[q][i][j]
            q +=1

    cached_Fq_Mq_vector = np.zeros([online_N,1])
    for q_f1 in range(q_f):
        for q_m1 in range(q_m):
            for i in range(online_N):
                delta = 2.

                cached_Fq_Mq_vector[i] += delta*theta_F(online_mu_parameters)[q_f1] \
                *theta_M(online_mu_parameters)[q_m1]*Fq_Mq_inner[q_f1][q_m1][i]

    cached_Aq_Mq_matrix = np.zeros([online_N, online_N])
    for q_a1 in range(q_a):
        for q_m1 in range(q_m):
            delta = 2.

            for i in range(online_N):
                for j in range(online_N):
                    cached_Aq_Mq_matrix[i,j] += delta*theta_A(online_mu_parameters)[q_a1] \
                    *theta_M(online_mu_parameters)[q_m1]*Aq_Mq_inner[q_a1][q_m1][i][j]

    cached_Mq_Mq_matrix = np.zeros([online_N, online_N])
    q=0
    for q_m1 in range(q_m):
        for q_m2 in range(q_m1, q_m):
            if(q_m1==q_m2):
                delta = 1.
            else:
                delta = 2.

            for i in range(online_N):
                for j in range(online_N):
                    cached_Mq_Mq_matrix[i,j] += delta*theta_M(online_mu_parameters)[q_m1] \
                    *theta_M(online_mu_parameters)[q_m2]*Mq_Mq_inner[q][i][j]
            q +=1

    return(cached_Fq_term, cached_Fq_Aq_vector, cached_Aq_Aq_matrix, cached_Fq_Mq_vector,
           cached_Aq_Mq_matrix, cached_Mq_Mq_matrix)

def compute_transient_residual_dual_norm(dt, euler_theta, current_timestep, online_N,
                                         RB_solution, old_RB_solution, cached_Fq_term,
                                         cached_Fq_Aq_vector, cached_Aq_Aq_matrix,
                                         cached_Fq_Mq_vector, cached_Aq_Mq_matrix,
                                         cached_Mq_Mq_matrix):

    """This method is responsible for computing the transient residual dual norm.

       Args:
           dt = time step size
           euler_theta = Time stepping scheme
           current_timestep = current time step
           RB_solution = reduced solution vector
           old_RB_solution = old reduced solution vector
           cached_Fq_term = Fq contribution to residual dual norm
           cached_Fq_Aq_vector = Fq-Aq contribution to residual dual norm
           cached_Aq_Aq_matrix = Aq-Aq contribution to residual dual norm
           cached_Fq_Mq_vector = Fq-Mq contribution to residual dual norm
           cached_Aq_Mq_matrix = Aq-Mq contribution to residual dual norm
           cached_Mq_Mq_matrix = Mq-Mq contribution to residual dual norm

        Returns:
           residual_norm_sq = the square root of the residual dual norm

    """
    current_control = get_control(current_timestep)

    RB_u_euler_theta = np.ndarray([online_N,1])
    mass_coeff = np.ndarray([online_N,1])

    RB_u_euler_theta = np.reshape((euler_theta*RB_solution)+((1.-euler_theta)*old_RB_solution)
                                  ,[online_N,1])
    mass_coeff = np.reshape((-(RB_solution-old_RB_solution)/dt),[online_N,1])

    residual_norm_sq = current_control*current_control*cached_Fq_term
    residual_norm_sq += current_control*np.dot(RB_u_euler_theta[:,0], cached_Fq_Aq_vector[:,0])
    residual_norm_sq += current_control*np.dot(mass_coeff[:,0], cached_Fq_Mq_vector[:,0])

    residual_norm_sq += np.sum(np.dot(RB_u_euler_theta,np.transpose(RB_u_euler_theta))
                               *cached_Aq_Aq_matrix)
    residual_norm_sq += np.sum(np.dot(mass_coeff,np.transpose(mass_coeff))
                               *cached_Mq_Mq_matrix)
    residual_norm_sq += np.sum(np.dot(RB_u_euler_theta,np.transpose(mass_coeff))
                               *cached_Aq_Mq_matrix)

    if(np.real(residual_norm_sq) < 0):
        residual_norm_sq = abs(residual_norm_sq)

    return(np.sqrt(residual_norm_sq))

def eval_output_dual_norm(output_id, q_l, online_mu_parameters, output_innerprods):
    """This method is responsible for evaluating the output dual norm.

       Args:
            output_id = the output for which the dual norm should be calcualted
            q_l = number of attached theta objects for each output vector
            online_mu_parameters = online parameters
            output_innerprods = the innerproducts from output-output

        Returns:
            output_bound_sq = the square root of the output dual norm

        """
    output_bound_sq = 0.
    q = 0
    for q_l1 in range(q_l):
        for q_l2 in range(q_l1, q_l):
            if(q_l1==q_l2):
                delta = 1.
            else:
                delta = 2.

            output_bound_sq += delta*np.real(np.conj(theta_O(online_mu_parameters)[output_id])
                                             * theta_O(online_mu_parameters)[output_id] *
                                             output_innerprods[output_id][q])

            q +=1

    return np.sqrt(output_bound_sq)

# def stability_lower_bound(online_mu_parameters):
#     return min(online_mu_parameters)
#
# def theta_A (online_mu_parameters):
#     return [online_mu_parameters[0], online_mu_parameters[1]]
#
# def theta_M (online_mu_parameters):
#     return [1.0]
#
# def theta_F (online_mu_parameters):
#     return [online_mu_parameters[0], online_mu_parameters[1]]
#
# def theta_O (online_mu_parameters):
#     return [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
#
# def theta_IC (online_mu_parameters):
#     return [1.0]
#
# def get_control(time_level):
#     return 1.0

def rb_solve_without_error_bound(online_mu_parameters, q_a, q_f, q_l, n_outputs, online_N,
                                 RB_Aq, RB_Fq, RB_Oq):
    """This method is responsible performing the rb solve without the output error bounds.

       Args:
           online_mu_parameters = online parameters
           q_a = number of stiffness matrices (A)
           q_f = number of load vectors (f)
           q_l = number of attached theta objects to each output vector
           n_outputs = number of output vectors (l)
           online_N = the number of basis functions that should be considered
           RB_Aq = reduced stiffness matrices
           RB_Fq = reduced load vectors
           RB_Oq = reduced load vectors

        Returns:
            RB_outputs = the output of interest

        """
    # assemble the RB system
    RB_system_matrix = np.sum(RB_Aq*theta_A(online_mu_parameters), axis = 2)

    # assemble the RB rhs
    RB_rhs = np.sum(RB_Fq*theta_F(online_mu_parameters), axis = 1)

    RB_solution = np.reshape(la.lu_solve(la.lu_factor(RB_system_matrix), RB_rhs),[online_N,1])

    # evaluate the RB outputs
    RB_outputs = np.sum(RB_solution*(RB_Oq[:,:,0]*theta_O(online_mu_parameters)), axis=0)

    return  (RB_outputs)

def rb_solve_with_error_bound(online_mu_parameters, q_a, q_f, q_l, n_outputs, online_N,
                              RB_Aq, RB_Fq, RB_Oq, Fq_inner, Fq_Aq_inner, Aq_Aq_inner,
                              output_inner):
    """This method is responsible performing the rb solve with the output error bounds.

       Args:
           online_mu_parameters = online parameters
           q_a = number of stiffness matrices (A)
           q_f = number of load vectors (f)
           q_l = number of attached theta objects to each output vector
           n_outputs = number of output vectors (l)
           online_N = the number of basis functions that should be considered
           RB_Aq = reduced stiffness matrices
           RB_Fq = reduced load vectors
           RB_Oq = reduced load vectors
           Fq_inner = the innerproducts from Fq-Fq
           Fq_Aq_inner = the innerproducts from Fq-Aq
           Aq_Aq_inner = the innerproducts from Aq-Aq
           output_inner = the innerproducts from output-output

        Returns:
            RB_outputs = the output of interest
            RB_output_error_bounds = the error bounds for the output of interest

        """
    # solution vector
    RB_solution=np.array([online_N,1])

    # assemble the RB system
    RB_system_matrix=np.zeros([online_N, online_N])
    RB_system_matrix = np.sum(RB_Aq*theta_A(online_mu_parameters), axis = 2)

    # assemble the RB rhs
    RB_rhs= np.zeros([online_N,1])
    RB_rhs = np.sum(RB_Fq*theta_F(online_mu_parameters), axis = 1)

    RB_solution = la.lu_solve(la.lu_factor(RB_system_matrix), RB_rhs)

    # evaluate the RB outputs and corresponding errors
    epsilon_N = compute_residual_dual_norm(Fq_inner, Fq_Aq_inner, Aq_Aq_inner, q_a, q_f,
                                               online_mu_parameters, online_N, RB_solution)
    alpha_LB = stability_lower_bound(online_mu_parameters)

    if(alpha_LB>0.):
        abs_error_bound = epsilon_N/alpha_LB
    else:
        print("The lower bound must be larger than 0.")
        return

    RB_outputs = np.zeros([n_outputs,1])
    RB_output_error_bounds = np.zeros([n_outputs,1])

    for i in range(n_outputs):
        RB_outputs[i] = np.sum(RB_Oq[:,i,:]*theta_O(online_mu_parameters)[i]*RB_solution)
        RB_output_error_bounds[i]=abs_error_bound* eval_output_dual_norm(i, q_l,
                                                                         online_mu_parameters,
                                                                         output_inner)

    return (RB_outputs, RB_output_error_bounds)


def transient_rb_solve_without_error_bound(online_mu_parameters, q_a, q_m, q_f,
                                           q_l, n_outputs, online_N, n_timesteps,
                                           dt, euler_theta, RB_Aq, RB_Mq, RB_Fq,
                                           RB_Oq, initial_conditions,
                                           parameter_dependent_IC = False, q_ic = 0,
                                           RB_L2_matrix = 0, varying_timesteps = False,
                                           growth_rate = 1.0, threshold = 1.0e30,
                                           time_dependent_parameters = False,
                                           ID_param = [0], start_time = 0.0,
                                           end_time = 0.0):

    """This method is responsible performing the transient rb solve without the
       output error bounds.

       Args:
           online_mu_parameters = online parameters
           q_a = number of stiffness matrices (A)
           q_m = number of mass matrices (M)
           q_f = number of load vectors (f)
           q_l = number of attached theta objects to each output vector
           n_outputs = number of output vectors (l)
           online_N = the number of basis functions that should be considered
           n_timesteps = number of time steps
           dt = time step size
           euler_theta = Time stepping scheme
           RB_Aq = reduced stiffness matrices
           RB_Mq = reduced mass matrices
           RB_Fq = reduced load vectors
           RB_Oq = reduced load vectors
           initial_conditions = initial conditions
           parameter_dependent_IC = determines whether the initial conditions
                                    are parameter dependent or note
           q_ic = number of intial conditions (IC)
           RB_L2_matrix = reduced L2 matrix

        Returns:
            RB_outputs_all_k = the output of interest for all timesteps

        """
    # assemble the mass matrix
    # RB_mass_matrix_N = np.zeros([online_N, online_N])
    if(time_dependent_parameters == True):
        time = 0;
        online_mu_parameters_initial = np.zeros([len(online_mu_parameters)])
        for i in range (len(online_mu_parameters)):
            online_mu_parameters_initial[i] = online_mu_parameters[i]
        online_mu_parameters = calculate_time_dependent_mu(online_mu_parameters,
                                                           online_mu_parameters_initial,
                                                           time, ID_param, dt,
                                                           start_time,end_time)

    RB_mass_matrix_N = np.sum(RB_Mq*theta_M(online_mu_parameters), axis = 2)

    # assemble LHS matrix
    # RB_LHS_matrix = np.zeros([online_N, online_N])
    RB_LHS_matrix = RB_mass_matrix_N * (1./dt)
    RB_LHS_matrix += np.sum(RB_Aq*theta_A(online_mu_parameters), axis = 2)

    # assemble RHS matrix
    # RB_RHS_matrix = np.zeros([online_N, online_N])
    RB_RHS_matrix = RB_mass_matrix_N * (1./dt)
    RB_RHS_matrix += np.sum(-(1.-euler_theta)*RB_Aq*theta_A(online_mu_parameters), axis = 2)

    # add forcing terms
    RB_RHS_save = np.zeros([online_N])
    RB_RHS_save += np.sum(RB_Fq*theta_F(online_mu_parameters), axis = 1)

    # add the intial conditions to the solution vector
    RB_solution = np.zeros([online_N,1])

    if(parameter_dependent_IC==False):
        RB_solution = initial_conditions
    else:
        # RB_rhs_N= np.zeros([online_N,1]);
        RB_rhs_N += initial_conditions*theta_IC(online_mu_parameters)
        RB_solution = la.lu_solve(la.lu_factor(RB_L2_matrix), RB_rhs_N)
        RB_solution = RB_solution[:,0]

    old_RB_solution = np.zeros([online_N,1])

    # initialize the RB rhs
    # RB_rhs = np.zeros([online_N,1])

    # initialize the vectors storing the solution data
    RB_temporal_solution_data = np.zeros([n_timesteps+1,online_N])

    # load the initial data
    RB_temporal_solution_data[0] = RB_solution

    # set outputs at initial time
    RB_outputs_all_k = np.zeros([n_outputs, n_timesteps+1])

    for i in range(n_outputs):
        RB_outputs_all_k[i][0] = np.sum(RB_Oq[:,i,:]*theta_O(online_mu_parameters)[i]*
                                        np.reshape(RB_solution, [online_N,1]))

    for i in range(1,n_timesteps+1):
        if(varying_timesteps==True and time_dependent_parameters == False):
            # assemble LHS matrix
            # RB_LHS_matrix = np.zeros([online_N, online_N])
            RB_LHS_matrix = RB_mass_matrix_N * (1./dt)
            RB_LHS_matrix += np.sum(RB_Aq*theta_A(online_mu_parameters), axis = 2)

            # assemble RHS matrix
            # RB_RHS_matrix = np.zeros([online_N, online_N])
            RB_RHS_matrix = RB_mass_matrix_N * (1./dt)
            RB_RHS_matrix += np.sum(-(1.-euler_theta)*RB_Aq*theta_A(online_mu_parameters), axis = 2)
        elif((varying_timesteps==False and time_dependent_parameters == True) or (varying_timesteps==True and time_dependent_parameters==True)):
            time +=dt
            online_mu_parameters = calculate_time_dependent_mu(online_mu_parameters,
                                                               online_mu_parameters_initial,
                                                               time, ID_param, dt,
                                                               start_time,end_time)
            RB_mass_matrix_N = np.sum(RB_Mq*theta_M(online_mu_parameters), axis = 2)

            # assemble LHS matrix
            # RB_LHS_matrix = np.zeros([online_N, online_N])
            RB_LHS_matrix = RB_mass_matrix_N * (1./dt)
            RB_LHS_matrix += np.sum(RB_Aq*theta_A(online_mu_parameters), axis = 2)

            # assemble RHS matrix
            # RB_RHS_matrix = np.zeros([online_N, online_N])
            RB_RHS_matrix = RB_mass_matrix_N * (1./dt)
            RB_RHS_matrix += np.sum(-(1.-euler_theta)*RB_Aq*theta_A(online_mu_parameters), axis = 2)

            # add forcing terms
            RB_RHS_save = np.zeros([online_N])
            RB_RHS_save += np.sum(RB_Fq*theta_F(online_mu_parameters), axis = 1)


        old_RB_solution = RB_solution

        RB_rhs = np.dot(RB_RHS_matrix, old_RB_solution)

        # add forcing term
        RB_rhs += get_control(i)*RB_RHS_save

        RB_solution = la.lu_solve(la.lu_factor(RB_LHS_matrix), RB_rhs)

        # Save RB_solution for current time level
        RB_temporal_solution_data[i] = RB_solution;

        for j in range(n_outputs):
            RB_outputs_all_k[j][i] = np.sum(RB_Oq[:,j,:]*theta_O(online_mu_parameters)[j]*
                                          np.reshape(RB_solution, [online_N,1]))

        if(dt < threshold):
            dt*=growth_rate

        if(time_dependent_parameters == True):
            for i in range (len(online_mu_parameters)):
                online_mu_parameters[i] = online_mu_parameters_initial[i]

    return (RB_outputs_all_k)


def transient_rb_state_solve(online_mu_parameters, q_a, q_m, q_f,
                                           q_l, n_outputs, online_N, n_timesteps,
                                           dt, euler_theta, RB_Aq, RB_Mq, RB_Fq,
                                           RB_Oq, initial_conditions,
                                           parameter_dependent_IC = False, q_ic = 0,
                                           RB_L2_matrix = 0, varying_timesteps = False,
                                           growth_rate = 1.0, threshold = 1.0e30,
                                           time_dependent_parameters = False,
                                           ID_param = [0], start_time = 0.0,
                                           end_time = 0.0):

    """This method is responsible performing the transient rb solve without the
       output error bounds.

       Args:
           online_mu_parameters = online parameters
           q_a = number of stiffness matrices (A)
           q_m = number of mass matrices (M)
           q_f = number of load vectors (f)
           q_l = number of attached theta objects to each output vector
           n_outputs = number of output vectors (l)
           online_N = the number of basis functions that should be considered
           n_timesteps = number of time steps
           dt = time step size
           euler_theta = Time stepping scheme
           RB_Aq = reduced stiffness matrices
           RB_Mq = reduced mass matrices
           RB_Fq = reduced load vectors
           RB_Oq = reduced load vectors
           initial_conditions = initial conditions
           parameter_dependent_IC = determines whether the initial conditions
                                    are parameter dependent or note
           q_ic = number of intial conditions (IC)
           RB_L2_matrix = reduced L2 matrix

        Returns:
            RB_solution = state

        """
    # assemble the mass matrix
    # RB_mass_matrix_N = np.zeros([online_N, online_N])
    if(time_dependent_parameters == True):
        time = 0;
        online_mu_parameters_initial = np.zeros([len(online_mu_parameters)])
        for i in range (len(online_mu_parameters)):
            online_mu_parameters_initial[i] = online_mu_parameters[i]
        online_mu_parameters = calculate_time_dependent_mu(online_mu_parameters,
                                                           online_mu_parameters_initial,
                                                           time, ID_param, dt,
                                                           start_time,end_time)

    RB_mass_matrix_N = np.sum(RB_Mq*theta_M(online_mu_parameters), axis = 2)

    # assemble LHS matrix
    # RB_LHS_matrix = np.zeros([online_N, online_N])
    RB_LHS_matrix = RB_mass_matrix_N * (1./dt)
    RB_LHS_matrix += np.sum(RB_Aq*theta_A(online_mu_parameters), axis = 2)

    # assemble RHS matrix
    # RB_RHS_matrix = np.zeros([online_N, online_N])
    RB_RHS_matrix = RB_mass_matrix_N * (1./dt)
    RB_RHS_matrix += np.sum(-(1.-euler_theta)*RB_Aq*theta_A(online_mu_parameters), axis = 2)

    # add forcing terms
    RB_RHS_save = np.zeros([online_N])
    RB_RHS_save += np.sum(RB_Fq*theta_F(online_mu_parameters), axis = 1)

    # add the intial conditions to the solution vector
    RB_solution = np.zeros([online_N,1])

    if(parameter_dependent_IC==False):
        RB_solution = initial_conditions
    else:
        # RB_rhs_N= np.zeros([online_N,1]);
        RB_rhs_N += initial_conditions*theta_IC(online_mu_parameters)
        RB_solution = la.lu_solve(la.lu_factor(RB_L2_matrix), RB_rhs_N)
        RB_solution = RB_solution[:,0]

    old_RB_solution = np.zeros([online_N,1])

    # initialize the RB rhs
    # RB_rhs = np.zeros([online_N,1])

    # initialize the vectors storing the solution data
    RB_temporal_solution_data = np.zeros([n_timesteps+1,online_N])

    # load the initial data
    RB_temporal_solution_data[0] = RB_solution

    for i in range(1,n_timesteps+1):
        if(varying_timesteps==True and time_dependent_parameters == False):
            # assemble LHS matrix
            # RB_LHS_matrix = np.zeros([online_N, online_N])
            RB_LHS_matrix = RB_mass_matrix_N * (1./dt)
            RB_LHS_matrix += np.sum(RB_Aq*theta_A(online_mu_parameters), axis = 2)

            # assemble RHS matrix
            # RB_RHS_matrix = np.zeros([online_N, online_N])
            RB_RHS_matrix = RB_mass_matrix_N * (1./dt)
            RB_RHS_matrix += np.sum(-(1.-euler_theta)*RB_Aq*theta_A(online_mu_parameters), axis = 2)
        elif((varying_timesteps==False and time_dependent_parameters == True) or (varying_timesteps==True and time_dependent_parameters==True)):
            time +=dt
            online_mu_parameters = calculate_time_dependent_mu(online_mu_parameters,
                                                               online_mu_parameters_initial,
                                                               time, ID_param, dt,
                                                               start_time,end_time)
            RB_mass_matrix_N = np.sum(RB_Mq*theta_M(online_mu_parameters), axis = 2)

            # assemble LHS matrix
            # RB_LHS_matrix = np.zeros([online_N, online_N])
            RB_LHS_matrix = RB_mass_matrix_N * (1./dt)
            RB_LHS_matrix += np.sum(RB_Aq*theta_A(online_mu_parameters), axis = 2)

            # assemble RHS matrix
            # RB_RHS_matrix = np.zeros([online_N, online_N])
            RB_RHS_matrix = RB_mass_matrix_N * (1./dt)
            RB_RHS_matrix += np.sum(-(1.-euler_theta)*RB_Aq*theta_A(online_mu_parameters), axis = 2)

            # add forcing terms
            RB_RHS_save = np.zeros([online_N])
            RB_RHS_save += np.sum(RB_Fq*theta_F(online_mu_parameters), axis = 1)


        old_RB_solution = RB_solution

        RB_rhs = np.dot(RB_RHS_matrix, old_RB_solution)

        # add forcing term
        RB_rhs += get_control(i)*RB_RHS_save

        RB_solution = la.lu_solve(la.lu_factor(RB_LHS_matrix), RB_rhs)

        if(dt < threshold):
            dt*=growth_rate

        if(time_dependent_parameters == True):
            for i in range (len(online_mu_parameters)):
                online_mu_parameters[i] = online_mu_parameters_initial[i]

    return (RB_solution)


def transient_rb_solve_with_error_bound(online_mu_parameters, q_a, q_m, q_f, q_l,
                                        n_outputs, online_N, n_timesteps, dt, euler_theta,
                                        RB_Aq, RB_Mq, RB_Fq, RB_Oq, Fq_inner, Fq_Aq_inner,
                                        Aq_Aq_inner, output_inner, Fq_Mq_inner, Aq_Mq_inner,
                                        Mq_Mq_inner, initial_conditions, initial_L2_error,
                                        parameter_dependent_IC = False, q_ic = 0,
                                        RB_L2_matrix = 0, varying_timesteps = False,
                                        growth_rate = 1.0, threshold = 1e30,
                                        time_dependent_parameters = False,
                                        ID_param = [0], start_time = 0.0, end_time = 0.0):

    """This method is responsible performing the transient rb solve with the
       output error bounds.

       Args:
           online_mu_parameters = online parameters
           q_a = number of stiffness matrices (A)
           q_m = number of mass matrices (M)
           q_f = number of load vectors (f)
           q_l = number of attached theta objects to each output vector
           n_outputs = number of output vectors (l)
           online_N = the number of basis functions that should be considered
           n_timesteps = number of time steps
           dt = time step size
           euler_theta = Time stepping scheme
           RB_Aq = reduced stiffness matrices
           RB_Mq = reduced mass matrices
           RB_Fq = reduced load vectors
           RB_Oq = reduced load vectors
           Fq_inner = the innerproducts from Fq-Fq
           Fq_Aq_inner = the innerproducts from Fq-Aq
           Aq_Aq_inner = the innerproducts from Aq-Aq
           output_inner = the innerproducts from output-output
           Fq_Mq_inner = the innerproducts from Fq-Mq
           Aq_Mq_inner = the innerproducts from Aq-Mq
           Mq_Mq_inner = the innerproducts from Mq-Mq
           initial_conditions = initial conditions
           initial_L2_error = initial L2 error

        Returns:
            RB_outputs_all_k = the output of interest for all timesteps
            RB_output_error_bounds_all_k = the error bounds for the output of interest

        """

    if(time_dependent_parameters == True):
        time = 0;
        online_mu_parameters_initial = np.zeros([len(online_mu_parameters)])
        for i in range (len(online_mu_parameters)):
            online_mu_parameters_initial[i] = online_mu_parameters[i]
        online_mu_parameters = calculate_time_dependent_mu(online_mu_parameters,
                                                           online_mu_parameters_initial,
                                                           time, ID_param, dt,
                                                           start_time,end_time)
    # assemble the mass matrix
    RB_mass_matrix_N = np.zeros([online_N, online_N])
    RB_mass_matrix_N = np.sum(RB_Mq*theta_M(online_mu_parameters), axis = 2)

    # assemble LHS matrix
    RB_LHS_matrix = np.zeros([online_N, online_N])
    RB_LHS_matrix = RB_mass_matrix_N * (1./dt)
    RB_LHS_matrix += np.sum(RB_Aq*theta_A(online_mu_parameters), axis = 2)

    # assemble RHS matrix
    RB_RHS_matrix = np.zeros([online_N, online_N])
    RB_RHS_matrix = RB_mass_matrix_N * (1./dt)
    RB_RHS_matrix += np.sum(-(1.-euler_theta)*RB_Aq*theta_A(online_mu_parameters), axis = 2)

    # add forcing terms
    RB_RHS_save = np.zeros([online_N])
    RB_RHS_save += np.sum(RB_Fq*theta_F(online_mu_parameters), axis = 1)

    # add the intial conditions to the solution vector
    RB_solution = np.zeros([online_N,1])

    if(parameter_dependent_IC==False):
        RB_solution = initial_conditions
    else:
        RB_rhs_N= np.zeros([online_N,1]);
        RB_rhs_N += initial_conditions*theta_IC(online_mu_parameters)
        RB_solution = la.lu_solve(la.lu_factor(RB_L2_matrix), RB_rhs_N)
        RB_solution = RB_solution[:,0]

    old_RB_solution = np.zeros([online_N,1])

    # initialize the RB rhs
    RB_rhs = np.zeros([online_N,1])

    # initialize the vectors storing the solution data
    RB_temporal_solution_data = np.zeros([n_timesteps+1,online_N])

    # load the initial data
    RB_temporal_solution_data[0] = RB_solution

    # set error bounds at initial time
    error_bound_sum = 0.
    alpha_LB = 0.
    error_bound_all_k = np.zeros([n_timesteps+1,1])

    error_bound_sum += initial_L2_error**2
    error_bound_all_k[0] = np.sqrt(error_bound_sum)

    # set outputs at initial time
    RB_outputs_all_k = np.zeros([n_outputs, n_timesteps+1])
    RB_output_error_bounds_all_k = np.zeros([n_outputs, n_timesteps+1])

    for i in range(n_outputs):
        RB_outputs_all_k[i][0] = np.sum(RB_Oq[:,i,:]*theta_O(online_mu_parameters)[i]*
                                        np.reshape(RB_solution, [online_N,1]))
        RB_output_error_bounds_all_k[i][0] = error_bound_all_k[0] * eval_output_dual_norm(i,
                                                                        q_l,
                                                                        online_mu_parameters,
                                                                        output_inner)


    if(time_dependent_parameters == False):
        alpha_LB = stability_lower_bound(online_mu_parameters)

        [cached_Fq_term, cached_Fq_Aq_vector, cached_Aq_Aq_matrix, cached_Fq_Mq_vector,\
         cached_Aq_Mq_matrix, cached_Mq_Mq_matrix] = cache_online_residual_terms(online_N,
                                                                             q_a, q_m, q_f,
                                                                             Fq_inner,
                                                                             Fq_Aq_inner,
                                                                             Aq_Aq_inner,
                                                                             Fq_Mq_inner,
                                                                             Aq_Mq_inner,
                                                                             Mq_Mq_inner,
                                                                    online_mu_parameters)


    for i in range(1,n_timesteps+1):
        if(varying_timesteps==True and time_dependent_parameters == False):
            # assemble LHS matrix
            # RB_LHS_matrix = np.zeros([online_N, online_N])
            RB_LHS_matrix = RB_mass_matrix_N * (1./dt)
            RB_LHS_matrix += np.sum(RB_Aq*theta_A(online_mu_parameters), axis = 2)

            # assemble RHS matrix
            # RB_RHS_matrix = np.zeros([online_N, online_N])
            RB_RHS_matrix = RB_mass_matrix_N * (1./dt)
            RB_RHS_matrix += np.sum(-(1.-euler_theta)*RB_Aq*theta_A(online_mu_parameters), axis = 2)
        elif((varying_timesteps==False and time_dependent_parameters == True) or (varying_timesteps==True and time_dependent_parameters==True)):
            time +=dt
            online_mu_parameters = calculate_time_dependent_mu(online_mu_parameters,
                                                               online_mu_parameters_initial,
                                                               time, ID_param, dt,
                                                               start_time,end_time)
            RB_mass_matrix_N = np.sum(RB_Mq*theta_M(online_mu_parameters), axis = 2)

            # assemble LHS matrix
            # RB_LHS_matrix = np.zeros([online_N, online_N])
            RB_LHS_matrix = RB_mass_matrix_N * (1./dt)
            RB_LHS_matrix += np.sum(RB_Aq*theta_A(online_mu_parameters), axis = 2)

            # assemble RHS matrix
            # RB_RHS_matrix = np.zeros([online_N, online_N])
            RB_RHS_matrix = RB_mass_matrix_N * (1./dt)
            RB_RHS_matrix += np.sum(-(1.-euler_theta)*RB_Aq*theta_A(online_mu_parameters), axis = 2)

            # add forcing terms
            RB_RHS_save = np.zeros([online_N])
            RB_RHS_save += np.sum(RB_Fq*theta_F(online_mu_parameters), axis = 1)

        old_RB_solution = RB_solution

        RB_rhs = np.dot(RB_RHS_matrix, old_RB_solution)

        # add forcing term
        RB_rhs += get_control(i)*RB_RHS_save

        RB_solution = la.lu_solve(la.lu_factor(RB_LHS_matrix), RB_rhs)

        # Save RB_solution for current time level
        RB_temporal_solution_data[i] = RB_solution;

        if(time_dependent_parameters == True or varying_timesteps == True):
            alpha_LB = stability_lower_bound(online_mu_parameters)

        if(time_dependent_parameters == True):
            [cached_Fq_term, cached_Fq_Aq_vector, cached_Aq_Aq_matrix, cached_Fq_Mq_vector,\
             cached_Aq_Mq_matrix, cached_Mq_Mq_matrix] = cache_online_residual_terms(online_N,
                                                                             q_a, q_m, q_f,
                                                                             Fq_inner,
                                                                             Fq_Aq_inner,
                                                                             Aq_Aq_inner,
                                                                             Fq_Mq_inner,
                                                                             Aq_Mq_inner,
                                                                             Mq_Mq_inner,
                                                                    online_mu_parameters)

        epsilon_N = compute_transient_residual_dual_norm(dt, euler_theta, i, online_N,
                                                         np.reshape(RB_solution, [online_N,1]),
                                                         np.reshape(old_RB_solution,
                                                                    [online_N,1]),
                                                         cached_Fq_term, cached_Fq_Aq_vector,
                                                         cached_Aq_Aq_matrix,
                                                         cached_Fq_Mq_vector,
                                                         cached_Aq_Mq_matrix,
                                                         cached_Mq_Mq_matrix)

        error_bound_sum += dt * (epsilon_N**2)
        error_bound_all_k[i] = np.sqrt(error_bound_sum/alpha_LB)

        for j in range(n_outputs):
            RB_outputs_all_k[j][i] = np.sum(RB_Oq[:,j,:]*theta_O(online_mu_parameters)[j]*
                                          np.reshape(RB_solution, [online_N,1]))
            RB_output_error_bounds_all_k[j][i] = error_bound_all_k[i] \
            * eval_output_dual_norm(j, q_l, online_mu_parameters, output_inner)

        if(dt < threshold):
            dt*=growth_rate

        if(time_dependent_parameters == True):
            for i in range (len(online_mu_parameters)):
                online_mu_parameters[i] = online_mu_parameters_initial[i]


    return (RB_outputs_all_k, RB_output_error_bounds_all_k)


def calculate_time_dependent_mu(online_mu_parameters, online_mu_parameters_initial,
                                time, ID_param, dt, start_time, end_time):

    pre_factor = 1.0
    if (time < start_time or time - dt >= end_time):
        pre_factor = 0.0
    elif (time - dt < start_time):
        if (time <= end_time):
            pre_factor *= (time - start_time) / dt
        else:
            pre_factor *= (end_time - start_time) / dt
    elif (time > end_time):
        pre_factor *= (end_time - (time - dt)) / dt

    for i in range(len(ID_param)):
        online_mu_parameters[ID_param[i]] = pre_factor * online_mu_parameters_initial[ID_param[i]]

    return online_mu_parameters
