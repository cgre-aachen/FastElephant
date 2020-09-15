#################################################################
# Model specific
def stability_lower_bound(online_mu_parameters):
    return min(online_mu_parameters)

def theta_A (online_mu_parameters):
    return [online_mu_parameters[0]]

def theta_F (online_mu_parameters):
        return [online_mu_parameters[1]]

def theta_O (online_mu_parameters):
    return [1.0]
#################################################################
