# Portfolio Optimization on a Quantum computer
Qiskit implementation of a Portfolio Optimization.

## Requirements
* Python 3.8+
* Qiskit
* CVXPY
* MOSEK
* tqdm

```shell
pip install -r requirements.txt
```

## VQE Solver
The VQE Solver takes a Covariance matrix and changes it to an Ising problem. The Ising model's hamiltonian is then minimized with the VQE.

```python
from qpo.vqe.vqe_solver import VQESolver

from qiskit import BasicAer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import SLSQP

vqe = VQESolver()
vqe.qp(Cov = Cov)
vqe.to_ising(Nq = Nq)

# Prepare QuantumInstance
qi = QuantumInstance(BasicAer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

# Select the VQE parameters
N = Cov.shape[0]
ansatz = TwoLocal(num_qubits=N*Nq, 
                    rotation_blocks=['ry','rz'], 
                    entanglement_blocks='cz',
                    reps=args.reps,
                    entanglement='full')
slsqp = SLSQP(maxiter=args.maxiter)
vqe.vqe_instance(ansatz=ansatz, optimizer=slsqp, quantum_instance=qi)

res_vqe = vqe.solve()
print(res_vqe)
```

## License
[Apache License 2.0](https://github.com/adelshb/quantum-porforlio-optimization-via-entanglement-forging/blob/main/LICENSE)