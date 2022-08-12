# -*- coding: utf-8 -*-
import numpy as np


def lorenzian(frequency, amplitude, center, sigma, offset):
    # http://openafox.com/science/peak-function-derivations.html
    return (amplitude / np.pi) * (
        sigma / ((frequency - center) ** 2 + sigma**2)
    ) + offset


def get_values(df, quantity, unit):
    return df[quantity].pint.to(unit).pint.magnitude
