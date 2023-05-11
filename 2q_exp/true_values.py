from copy import deepcopy

import numpy as np

Rx_true = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)
Ry_true = np.array([[1, -1], [1, 1]]) / np.sqrt(2)

Id_true = np.eye(16, dtype=complex)
Rx0_true = np.kron(Rx_true, np.eye(2, dtype=complex))
Rx0_true = np.kron(Rx0_true, Rx0_true.conj())

Ry0_true = np.kron(Ry_true, np.eye(2, dtype=complex))
Ry0_true = np.kron(Ry0_true, Ry0_true.conj())

Rx1_true = np.kron(np.eye(2, dtype=complex), Rx_true)
Rx1_true = np.kron(Rx1_true, Rx1_true.conj())

Ry1_true = np.kron(np.eye(2, dtype=complex), Ry_true)
Ry1_true = np.kron(Ry1_true, Ry1_true.conj())

CZ_true = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex
)
CZ_true = np.kron(CZ_true, CZ_true.conj())

X_true = np.array(
    [Id_true, Rx0_true, Ry0_true, Rx1_true, Ry1_true, CZ_true, deepcopy(Id_true)]
)


E_true = []
for i in range(2):
    for j in range(2):
        E = np.zeros((4, 4), dtype=complex)
        E[i * 2 + j][i * 2 + j] = 1
        E_true.append(
            E.reshape(
                -1,
            ).tolist()
        )
E_true = np.array(E_true, dtype=complex)


rho_true = [1] + [0] * 15
rho_true = np.array(rho_true, dtype=complex)
