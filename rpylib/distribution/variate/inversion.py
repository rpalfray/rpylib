"""Inversion method to generate random variate from discrete probability distribution

"""

from bisect import bisect_left
from collections import deque
from collections.abc import Callable

from ..pairing import StatesManager
from ..sampling import Sampling
from ..univariate.uniform import Uniform


class InversionMethod(Sampling):
    """Inversion method

    The inversion method might be slower than other methods but one of the benefits is that there is no pre-computation
    and therefore no (much) impact on memory
    """
    def __init__(self, probability_to_jump_to_state: Callable[[tuple[int, ...]], float], state_manager: StatesManager):
        super().__init__()
        self.state_manager = state_manager
        self.probability_to_jump_to_state = probability_to_jump_to_state

        self.uniform = Uniform()
        state_increments0, _ = state_manager.project_index_to_state_increment(0)
        p_0 = probability_to_jump_to_state(state_increments0)
        self._cumulative_probabilities = deque([p_0])
        self._simulated_state_increments = deque([state_increments0])
        self._max_storage = 1_000_000

    def cost(self) -> int:
        return self.uniform.cost()

    def reset_sampling_cost(self):
        self.uniform.reset_sampling_cost()

    def sample(self, size: int = 1) -> list[tuple[int, ...]]:
        res = [self.sample_with_u(u) for u in self.uniform.sample(size=size)]
        return res

    def sample_with_u(self, u: float) -> tuple[int, ...]:
        cum_probabilities = self._cumulative_probabilities
        simulated_state_increments = self._simulated_state_increments
        state_increment = None

        if u > (s := cum_probabilities[-1]):
            x = len(cum_probabilities) - 1
            project_index_to_state_increment = self.state_manager.project_index_to_state_increment
            probability_to_jump_to_state = self.probability_to_jump_to_state
            while u > s:
                x += 1
                state_increment, break_here = project_index_to_state_increment(x, self._max_storage)
                if break_here:
                    break
                else:
                    s += probability_to_jump_to_state(state_increment)

                    if len(cum_probabilities) < self._max_storage:
                        cum_probabilities.append(s)
                        simulated_state_increments.append(state_increment)
        else:
            x = bisect_left(cum_probabilities, u)
            state_increment = simulated_state_increments[x]

        return state_increment
