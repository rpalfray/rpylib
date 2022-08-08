"""
Factorisation of code common to the both the 1d and Lévy copula case for coupling processes
"""

import numpy as np


def create_build_finer_grid_fun(epsilon: float, maturity: float):
    def _build_finer_grid_default(
        self, jump_times, fines_states_values, coarse_states_values
    ):
        return jump_times, fines_states_values, coarse_states_values

    def _build_finer_grid(self, jump_times, fines_states_values, coarse_states_values):
        dts = np.concatenate(([jump_times[0]], np.diff(jump_times)))
        if not any(dts > epsilon):
            return jump_times, fines_states_values, coarse_states_values
        else:
            positions = np.nonzero(dts > epsilon)[0]
            aug_fine_js = fines_states_values
            aug_coarse_js = coarse_states_values
            aug_dts = dts
            while positions.size > 0:
                aug_dts[positions] -= epsilon
                aug_dts = np.insert(aug_dts, positions, epsilon)
                aug_fine_js = np.insert(
                    aug_fine_js,
                    positions,
                    np.where(positions == 0, 0, aug_fine_js[..., positions - 1]),
                    axis=-1,
                )
                aug_coarse_js = np.insert(
                    aug_coarse_js,
                    positions,
                    np.where(positions == 0, 0, aug_coarse_js[..., positions - 1]),
                    axis=-1,
                )
                positions = np.nonzero(aug_dts > epsilon)[0]
            aug_jump_times = np.cumsum(aug_dts)

            return aug_jump_times, aug_fine_js, aug_coarse_js

    return _build_finer_grid_default if epsilon >= maturity else _build_finer_grid
