# The exercise involves solving a discretized wave equation in 1d with absorbing
# boundary layers (known as perfectly matched layers [PML]). The finite-difference
# equation for this example can be found in the attached write up [third equation
# from the top of page 3 in Section 4 "Finite-difference method", right after the
# sentence "In particular, the equations become:"]. Use the following parameters:
# \epsilon=1 everywhere, \omega = 2*\pi, PML length = 1 on both sides, \sigma_n =
# \sigma_0*u^2 [i.e., quadratic turn on, u is [0,1] within the right PML region,
# [1,0] in the left], \sigma_0 = 12, length of non-PML region = 10 [total length
# of computational cell=1+10+1=12], resolution=10 pixels/unit length (total number
# of pixels, N=10*12=120), and J_n=1 at the middle of the computational cell and
# zero everywhere else. This amounts to solving a linear equation of the form DE=J
# where D is the finite-difference operator (NxN matrix), J is the current source
# (Nx1 column vector), and E (Nx1 column vector) are the unknowns.

# We want to see a plot of the electric field in the computational cell as well as
# the code used to generate these results. Try to optimize for performance. Let me
# know if you have any additional questions.

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags


epsilon = 1  # everywhere
omega = 2 * np.pi
pml_length = 1  # on both sides
sigma_0 = 12
# sigma_n = sigma_0 * u^2  # [i.e., quadratic turn on, u is [0,1] within the right PML region, [1,0] in the left]
non_pml_region_len = 10  # [total length of computational cell=1+10+1=12]
resolution = 10  # pixels/unit length (total number of pixels, N=10*12=120)
J_n = 1  # at the middle of the computational cell and zero everywhere else.
N = 12

# (i * omega - sigma_n)J_n = epsilon_n(i * omega - sigma_n)^2 * E_n
#                            - ((E_n_plus_1 + E_n_minus_1 - 2 * E_n) / resolution^2)
#                            - ((E_n_plus_1 - E_n_minus_1) / (2 * resolution))
#                            * (sigma_x / (i * omega - sigma_n))


def plot(e):
    plt.plot(e)
    plt.ylabel('E(x)')
    plt.xlabel('x')
    plt.show()


def main():

    D = np.matrix(np.zeros([N + 1, N + 1]))
    j = np.zeros(N + 1)
    j.shape = (N + 1, 1)
    j[N // 2] = 1
    e = np.transpose(np.zeros(N))

    sigma = np.transpose(np.zeros(N + 1))
    sigma[0] = 12
    sigma[-1] = 12

    a = np.array([-4 * np.pi**2 + 0.02] * (N + 1))
    b = np.array([-0.01] * N)

    data = [a.tolist(), b.tolist(), b.tolist()]
    D = diags(data, [0, 1, -1], (N + 1, N + 1))
    # TODO: fix intial and final rows
    D = np.matrix(D.toarray())
    e = D.I * j
    # print(e)
    plot(e)


if __name__ == '__main__':
    main()
