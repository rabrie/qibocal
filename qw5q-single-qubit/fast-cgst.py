import argparse
import json
import os
import time
import numpy as np
from datetime import datetime
from qibolab import Platform
from qibolab.pulses import PulseSequence


parser = argparse.ArgumentParser()
parser.add_argument("--platform", type=str, default="qw5q_gold_qblox")
parser.add_argument("--qubit", type=int, default=2)
parser.add_argument("--depth", type=int, default=7)
parser.add_argument("--nsequences", type=int, default=10)
parser.add_argument("--nshots", type=int, default=10000)


def idx_to_sequence(platform, qubit, sequences_idx):
    """Converts random indices to a ``PulseSequence``.
    
    Pulses are sampled from the following gate set:
    {Idle = 0, pi/2 x-rotation = 1, pi/2 y-rotation = 2}.
    """
    duration = platform.create_RX_pulse(qubit, start=0).duration    
    
    ro_pulses = []
    start = 0
    sequence = PulseSequence()
    for idx in sequences_idx:
        for i in idx:
            if i == 1:
                sequence.add(platform.create_RX90_pulse(qubit, start=start))
            elif i == 2:
                sequence.add(platform.create_RX90_pulse(qubit, start=start, relative_phase=np.pi / 2))
            elif i != 0:
                raise IndexError
            start += duration
        ro_pulse = platform.create_MZ_pulse(qubit, start=start)
        ro_pulses.append(ro_pulse)
        sequence.add(ro_pulse)
        # wait for the qubit to relax between different sequences
        start += platform.relaxation_time

    return sequence, ro_pulses


def main(platform, qubit, nsequences=10, depth=7, nshots=10000):
    platform = Platform(platform)

    platform.connect()
    platform.setup()
    platform.start()

    # generate filename for data saving
    date = datetime.now().strftime("%Y-%m-%d")
    num = 0
    filename = f"fast-{platform.name}-{date}-000.json"
    while os.path.exists(filename):
        num += 1
        filename = f"fast-{platform.name}-{date}-{str(num).rjust(3, '0')}.json"

    # generate random indices for the gate set selections
    sequences_idx = np.random.randint(0, 3, size=(nsequences, depth)).astype(int)
    start_time = time.time()
    sequence, ro_pulses = idx_to_sequence(platform, qubit, sequences_idx)
    creation_time = time.time() - start_time

    # log metadata in json that will be saved
    data = {"platform": platform.name,
            "qubit": qubit,
            "depth": depth,
            "nsequences": nsequences,
            "nshots": nshots,
            "sequence_creation_time": creation_time,
            "execution_time": 0,
            "sequences": [],
            "hardware_probabilities": []}
    
    # execute pulse sequence
    start_time = time.time()
    results = platform.execute_pulse_sequence(sequence, nshots=nshots)
    execution_time = time.time() - start_time

    platform.stop()
    platform.disconnect()

    # save data
    data["execution_time"] = execution_time
    for idx, ro_pulse in zip(sequences_idx, ro_pulses):
        p = np.sum(results[ro_pulse.serial].shots) / nshots
        probs = [1 - p, p]
        data["sequences"].append([int(i) for i in idx])
        data["hardware_probabilities"].append(probs)

    # dump saved data to disk
    with open(filename, "w") as file:
        json.dump(data, file)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)