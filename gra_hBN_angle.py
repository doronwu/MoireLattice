# =============================================================================
# Graphene-hBN Continuum Model Band Structure Generator
# 
# Based on: Moon, P., & Koshino, M. (2014).
# Electronic properties of graphene/hexagonal-boron-nitride moiré superlattice.
# Phys. Rev. B, 90, 155406. https://doi.org/10.1103/PhysRevB.90.155406
#
# This script computes the valley-resolved moiré band structure of monolayer 
# graphene on hBN using a plane-wave expansion method. The model includes:
#   - Valley-dependent Dirac Hamiltonian for graphene
#   - Valley-dependent interlayer coupling (U1, U2, U3)
#   - On-site potential from hBN substrate
#
# Output: Energy bands for valley = ±1, plotted along high-symmetry path
#         (X → K → Y → X) in the moiré Brillouin zone.
#
# Parameters:
#   angle (deg):    twist angle between graphene and hBN
#   m_grid:         cutoff for plane wave basis
#   kd:             number of k-points per segment
#   Ew:             energy window for the plot
#
# Author: [Doron Wu]
# Date: [2025-04-03]
# =============================================================================


import numpy as np 
import matplotlib.pyplot as plt
# Constants and parameters
a = 2.46e-10
m_grid = 2
kd = 100
angle = 5
Ew = 1.4
# Rotation matrix
def rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

# Moire lattice vectors
pref = np.eye(2) - 55/56 * rot(angle * np.pi / 180)
m_b1 = 2 * np.pi / 3 / a * 3**0.5 * pref @ rot(np.pi/6) @ np.array([[1], [-np.sqrt(3)]])
m_b2 = rot(2*np.pi/3) @ m_b1

# High symmetry point
Gamma = np.array([[0.0], [0.0]])

# BN on-site potential
H_bn = np.array([[-1.4, 0], [0, 3.34]])  # eV

# Monolayer graphene Dirac Hamiltonian
def H_g(k, valley):
    kx, ky = k
    const = 0.8e6 * 6.582e-16
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    return const * (valley * sigma_x * kx + sigma_y * ky)

# Plane wave basis
G_list = [np.array([n, m]) for n in range(-m_grid, m_grid + 1) for m in range(-m_grid, m_grid + 1)]
N = len(G_list)

# Main loop over valleys
E_all = {}
for valley in [+1, -1]:
    # Interlayer coupling matrices (valley-dependent)
    U_1 = 0.152 * np.ones((2, 2))
    U_2 = 0.152 * np.array([[1, np.exp(-valley*2j / 3 * np.pi)], [np.exp(valley*2j / 3 * np.pi), 1]])
    U_3 = 0.152 * np.array([[1, np.exp(valley*2j / 3 * np.pi)], [np.exp(-valley*2j / 3 * np.pi), 1]])

    if valley == 1:
        Kp_moire = 1 / 3**0.5 * (rot(np.pi/2) @ m_b1)
        K_moire = 1 / 3**0.5 * (rot(np.pi/6) @ m_b1)
    else:
        Kp_moire = 1 / 3**0.5 * (rot(np.pi/2 + 2*np.pi/3) @ m_b1)
        K_moire = 1 / 3**0.5 * (rot(np.pi/6 + 2*np.pi/3) @ m_b1)

    K_point = [0, K_moire, Kp_moire]

    if valley == 1:
        kloop1 = [(1 - t) * Kp_moire + t * Gamma for t in np.linspace(0, 1, kd, endpoint=False)]
        kloop2 = [(1 - t) * Gamma + t * K_moire for t in np.linspace(0, 1, kd, endpoint=False)]
        kloop3 = [(1 - t) * K_moire + t * Kp_moire for t in np.linspace(0, 1, kd, endpoint=False)]
    else:
        kloop1 = [(1 - t) * K_moire + t * Kp_moire for t in np.linspace(0, 1, kd, endpoint=False)]
        kloop2 = [(1 - t) * Kp_moire + t * Gamma for t in np.linspace(0, 1, kd, endpoint=False)]
        kloop3 = [(1 - t) * Gamma + t * K_moire for t in np.linspace(0, 1, kd, endpoint=False)]

    kloop = np.array(kloop1 + kloop2 + kloop3)

    H_bn_exp = np.kron(np.eye(N), H_bn)
    energies = []

    for idx_k, k in enumerate(kloop):
        T_moire = np.zeros((2*N, 2*N), dtype=complex)
        for i, Gi in enumerate(G_list):
            for j, Gj in enumerate(G_list):
                delta_index = Gj - Gi
                if np.allclose(delta_index, [0,0], atol=1e-6):
                    T_moire[2*i:2*i+2, 2*j:2*j+2] = U_1
                elif np.allclose(delta_index, [valley*1, 0], atol=1e-6):
                    T_moire[2*i:2*i+2, 2*j:2*j+2] = U_2
                elif np.allclose(delta_index, [valley*1, valley*1], atol=1e-6):
                    T_moire[2*i:2*i+2, 2*j:2*j+2] = U_3

        H_g_matrix = np.zeros((2 * N, 2 * N), dtype=complex)
        for i in range(N):
            delta_k = k + G_list[i][0]*m_b1 + G_list[i][1]*m_b2 - K_point[valley]
            H_block = H_g(delta_k.flatten(), valley)
            H_g_matrix[2*i:2*i+2, 2*i:2*i+2] = H_block

        H_k = np.block([
            [H_g_matrix, T_moire.T.conj()],
            [T_moire, H_bn_exp]
        ])
        E_temp, _ = np.linalg.eigh(H_k)
        energies.append(E_temp)

    E_all[valley] = np.array(energies)

#Plotting
plt.figure(figsize=(4,6))
plt.ylim(-Ew, Ew)
for i in range(E_all[1].shape[1]):
    plt.plot(np.arange(len(kloop)), E_all[1][:,i]-0.042, c="b", ls="--")
    plt.plot(np.arange(len(kloop)), E_all[-1][:,i]-0.042, c="r")
plt.xticks([0, kd, 2*kd, 3*kd], ["X\nY'", "K\nX'", "Y\nK'", "X\nY'"])
plt.xlim(0, 3*kd)
plt.vlines([kd, 2*kd], -1.5, 1.5, colors="black", alpha=0.1, linestyle="--")
plt.hlines(0, 0, 3*kd-1, colors="black", alpha=0.1, linestyle="--")
plt.title("Graphene-hBN Continuum Model θ={} Deg".format(angle))
plt.ylabel("Energy - Dirac (eV)")
plt.tight_layout()
plt.show()
