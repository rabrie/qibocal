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


def idx_to_sequence(platform, qubit, idx):
    """Converts random indices to a ``PulseSequence``.
    
    Pulses are sampled from the following gate set:
    {Idle = 0, pi/2 x-rotation = 1, pi/2 y-rotation = 2}.
    """
    duration = platform.create_RX_pulse(qubit, start=0).duration    
    sequence = PulseSequence()
    start = 0
    for i in idx:
        if i == 1:
            sequence.add(platform.create_RX90_pulse(qubit, start=start))
        elif i == 2:
            sequence.add(platform.create_RX90_pulse(qubit, start=start, relative_phase=np.pi / 2))
        elif i != 0:
            raise IndexError
        start += duration
    sequence.add(platform.create_MZ_pulse(qubit, start=start))
    return sequence


def main(platform, qubit, nsequences=10, depth=7, nshots=10000):
    platform = Platform(platform)

    platform.connect()
    platform.setup()
    platform.start()

    # log metadata in json that will be saved
    data = {"platform": platform.name,
            "qubit": qubit,
            "depth": depth,
            "nsequences": nsequences,
            "nshots": nshots,
            "execution_times": [],
            "sequences": [],
            "hardware_probabilities": []}

    # generate filename for data saving
    date = datetime.now().strftime("%Y-%m-%d")
    num = 0
    filename = f"{platform.name}-{date}-000.json"
    while os.path.exists(filename):
        num += 1
        filename = f"{platform.name}-{date}-{str(num).rjust(3, '0')}.json"

    # generate random indices for the gate set selections
    sequences_idx = np.random.randint(0, 3, size=(nsequences, depth)).astype(int)
    
    for idx in sequences_idx:
        # execute pulse sequence
        start_time = time.time()
        sequence = idx_to_sequence(platform, qubit, idx)
        results = platform.execute_pulse_sequence(sequence, nshots=nshots)
        execution_time = time.time() - start_time
        p = np.sum(results[qubit].shots) / nshots
        probs = [1 - p, p]

        # save data
        data["execution_times"].append(execution_time)
        data["sequences"].append([int(i) for i in idx])
        data["hardware_probabilities"].append(probs)

        # dump saved data to disk
        with open(filename, "w") as file:
            json.dump(data, file)

    platform.stop()
    platform.disconnect()


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)