import online_stage as online

# Define the model specific decomposition
class model():
    def stability_lower_bound(self, online_mu_parameters):
        return min(online_mu_parameters)

    def theta_A (self,online_mu_parameters):
        return [online_mu_parameters[0]]

    def theta_F (self,online_mu_parameters):
        return [online_mu_parameters[1]]
    def theta_O (self,online_mu_parameters):
        return [1.0]

# Define the model specific input parameters for the reduced model
path_offline_data = "offline_data"
q_a = 1
q_f = 1
q_l = 1
n_outputs = 1
online_mu_parameters = np.array([2.5 3.0])

# Read the offline Data
[online_N,RB_Aq,RB_Fq, RB_Oq] = OnlineStage().read_offline_data(path_offline_data,
                                                        q_a, q_f, q_l, n_outputs)

# Perform the online solve
RB_outputs = model().rb_solve_without_error_bound(online_mu_parameters,q_a,
                                                q_f, q_l, n_outputs, online_N,
                                                RB_Aq, RB_Fq, RB_Oq)
