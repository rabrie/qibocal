import numpy as np
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence

from qibocal import plots
from qibocal.data import DataUnits
from qibocal.decorators import plot


@plot("Qubit States", plots.qubit_states)
def calibrate_qubit_states(
    platform: AbstractPlatform,
    qubits: list,
    nshots,
    points=10,
):
    platform.reload_settings()

    # create exc sequence
    state0_sequence = PulseSequence()
    state1_sequence = PulseSequence()

    RX_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        RX_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX_pulses[qubit].duration
        )

        state0_sequence.add(ro_pulses[qubit])
        state1_sequence.add(RX_pulses[qubit])
        state1_sequence.add(ro_pulses[qubit])

    data = DataUnits(name="data", options=["qubit", "iteration", "state"])

    state0_results = platform.execute_pulse_sequence(state0_sequence, nshots=nshots)
    for qubit in qubits:
        msr, phase, i, q = state0_results["demodulated_integrated_binned"][
            ro_pulses[qubit].serial
        ]
        results = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[rad]": phase,
            "qubit": [qubit] * nshots,
            "iteration": np.arange(nshots),
            "state": [0] * nshots,
        }
        data.add_data_from_dict(results)

    state1_results = platform.execute_pulse_sequence(state1_sequence, nshots=nshots)
    for qubit in qubits:
        msr, phase, i, q = state1_results["demodulated_integrated_binned"][
            ro_pulses[qubit].serial
        ]
        results = {
            "MSR[V]": msr,
            "i[V]": i,
            "q[V]": q,
            "phase[rad]": phase,
            "qubit": [qubit] * nshots,
            "iteration": np.arange(nshots),
            "state": [1] * nshots,
        }
        data.add_data_from_dict(results)

    yield data
