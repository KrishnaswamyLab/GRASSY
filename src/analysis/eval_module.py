import torch
import rdkit

# TODO: create an evaluation key in the yaml configs to control which metrics to compute

"""
NOTE:
Check EDM's QM9 generated molecule evaluator from here:
https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/qm9/rdkit_functions.py
"""

class GRASSYEvalModule():
    def __init__(self):
        pass

    def evaluate(self) -> dict:
        """
        Runs all evaluation functions on generated molecules.

        :return: dictionary containing final evaluation metrics.
        """
        return {}