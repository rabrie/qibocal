import json
import numpy as np
from qibo.models import Circuit
from qibo import gates
from collections import defaultdict
from qibo import gates
from qibo.config import log

from qibolab import Platform
from qibolab.backends import QibolabBackend
from qibolab.paths import qibolab_folder
from qibolab.pulses import Pulse, FluxPulse, PulseSequence, ReadoutPulse, Exponential, Rectangular
from qibolab.platforms.abstract import AbstractPlatform
import time
import shutil

int_to_gate = {
    0: lambda q: gates.I(*q),
    1: lambda q: gates.RX(q[0], np.pi / 2),
    2: lambda q: gates.RY(q[0], np.pi / 2),
    3: lambda q: gates.RX(q[1], np.pi / 2),
    4: lambda q: gates.RY(q[1], np.pi / 2),
    5: lambda q: gates.CZ(*q),
}

def circuit_to_sequence(platform: AbstractPlatform, nqubits, qubits, circuit):
    # Define PulseSequence
    sequence = PulseSequence()
    virtual_z_phases = defaultdict(int)

    for index in circuit:
        # Single qubits gates except Id
        if 0 < index < 5:
            qubit = qubits[0] if index < 3 else qubits[1]
            phase = 0 if index % 2 else np.pi / 2
            sequence.add(
                platform.create_RX90_pulse(
                    qubit,
                    start=sequence.finish,
                    # start=sequence.get_qubit_pulses(qubit).finish,
                    relative_phase=virtual_z_phases[qubit]+phase,
                )
            )
        # CZ gate
        elif index == 5:
            # determine the right start time based on the availability of the qubits involved
            # cz_qubits = {*cz_sequence.qubits, *self.qubits}
            cz_start = sequence.finish

            # create CZ pulse sequence with start time = 0
            (cz_sequence, cz_virtual_z_phases) = platform.create_CZ_pulse_sequence(qubits, cz_start)

            # add pulses to the sequence
            sequence.add(cz_sequence)

            # update z_phases registers
            for qubit in cz_virtual_z_phases:
                virtual_z_phases[qubit] += cz_virtual_z_phases[qubit]

    # Add measurement pulse
    measurement_start = sequence.finish

    for qubit in qubits:
        MZ_pulse = platform.create_MZ_pulse(qubit, start=measurement_start)
        sequence.add(MZ_pulse)

    return sequence


def workshop(num=6, depth=10):
    return list(np.random.randint(0, num, size=depth))


def calculate_probabilities(result1, result2):
    """Calculates two-qubit outcome probabilities from individual shots."""
    shots = np.stack([result1.shots, result2.shots]).T
    values, counts = np.unique(shots, axis=0, return_counts=True)
    nshots = np.sum(counts)
    return {f"{int(v1)}{int(v2)}": cnt / nshots for (v1, v2), cnt in zip(values, counts)}


nqubits = 5
qubits = [2, 3]
nshots = 10000
nsequences = 500
depths = [5, 7, 10]
# Define platform and load specific runcard
runcard = "qibolab/src/qibolab/runcards/qw5q_gold_qblox.yml"
timestr = time.strftime("%Y%m%d-%H%M")
shutil.copy(runcard, f"{timestr}_runcard.yml")

platform = Platform("qblox", runcard)

platform.connect()
platform.setup()
platform.start()

start_time = time.time()
execution_number = 0
for depth in depths:
    data = {"nqubits": 5, 
            "qubits": qubits, 
            "nshots": nshots, 
            "nsequences": nsequences,
            "depth": depth,
            "measurements": []}

    for _ in range(nsequences):
        execution_number += 1
        if execution_number % 30 == 0:
            log.info(f"execution munber {execution_number}, circuit depth {depth}")
            time_elapsed = time.time()-start_time
            total_number_executions = len(depths)*nsequences
            remaining_time = time_elapsed*total_number_executions/execution_number-time_elapsed
            log.info(f"estimated time to completion {int(remaining_time)//60}m {int(remaining_time) % 60}s")
        
        c = workshop(num=6, depth=depth)
        sequence = circuit_to_sequence(platform, nqubits, qubits, c)
        results = platform.execute_pulse_sequence(sequence, nshots=nshots)
        probs = calculate_probabilities(results[qubits[0]], results[qubits[1]])

        circuit_qibo = Circuit(nqubits)
        circuit_qibo.add([int_to_gate[i](qubits) for i in c])
        circuit_qibo.add(gates.M(*qubits))
        result = circuit_qibo(nshots=nshots)
        sim_probs = {k: float(v / nshots) for k, v in result.frequencies().items()}

        #print(c)
        #print(probs)
        #print(sim_probs)
        #print()
        data["measurements"].append({
            "circuit": [int(x) for x in c],
            "hardware_probabilities": probs,
            "simulation_probabilities": sim_probs
        })

        with open(f"{timestr}_cgst_depth_{depth}.json", "w") as file:
            json.dump(data, file)


platform.stop()
platform.disconnect()