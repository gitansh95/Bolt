"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, p1, p2, p3, params):

    q10 = params.q10
    q20 = params.q20

    p10 = params.p10
    p20 = params.p20

    sigma_q = params.sigma_q
    sigma_p = params.sigma_p

    q_profile = (1 / sigma_q**2 / (2 * np.pi)) * \
                af.exp(-0.5 * ((q1 - q10)**2 + (q2 - q20)**2) / sigma_q**2)
    p_profile = (1 / sigma_p**2 / (2 * np.pi)) * \
                af.exp(-0.5 * ((p1 - p10)**2 + (p2 - p20)**2) / sigma_p**2)
    
    f = q_profile * p_profile

    af.eval(f)
    return (f)
