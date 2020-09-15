import online_stage_dakota as online
import numpy as np
#**********************PART TO MODIFY**********************************#
q_a = 1
q_f = 1
q_l = 1
n_outputs = 1
path_offline_data = "offline_data"
[online_N,RB_Aq,RB_Fq, RB_Oq] = online.read_offline_data(path_offline_data,
                                                  q_a, q_f, q_l, n_outputs)
#**********************PART TO MODIFY**********************************#

def DwarfElephant_online_stage(**kwargs):
    # get online parameters
    x = kwargs['cv']

    online_mu_parameters = [x[0], x[1]]

    RB_output = online.rb_solve_without_error_bound(online_mu_parameters, q_a,
                                                    q_f, q_l, n_outputs, online_N,
                                                    RB_Aq, RB_Fq, RB_Oq)

    return RB_output
