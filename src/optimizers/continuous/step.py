from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .base import IOptimizer
from .local import local_perturb_optim
from ..core import InputVariables, OptimizerResult
from ..core.base import GoalFcn, InputArguments, IOptimizerConfig, StopReason
from ..core.types import AF
from ..solution_deck import SolutionDeck


@dataclass
class StepWiseOptimizerConfig(IOptimizerConfig):
    optimize_whole_solution_deck: bool = False
    max_perturbation: float = 0.1  # Fraction of domain
    pass


class StepWiseOptimizer(IOptimizer):
    def __init__(
        self,
        *,
        config: IOptimizerConfig,
        fcn: GoalFcn,
        variables: InputVariables,
        args: InputArguments | None = None,
        existing_soln_deck: SolutionDeck | None = None,
    ):
        super().__init__(
            config=config,
            fcn=fcn,
            variables=variables,
            args=args,
            existing_soln_deck=existing_soln_deck,
        )
        self.config: StepWiseOptimizerConfig = StepWiseOptimizerConfig(
            **{**config.__dict__}
        )

    def solve(self, *, preserve_percent: float = 0.0) -> OptimizerResult:
        best_soln_value: list[float] = list()
        # Start with the initial value from the input variables, and stepwise refine solve
        if not self.config.optimize_whole_solution_deck:
            x0 = np.array([v.initial_random_value(0.0) for v in self.variables])
            stop_reason: StopReason = "max_iterations"
            for gen in tqdm(
                range(self.config.num_generations),
                desc="Stepwise optimization generations",
            ):
                x0, x0_val = local_perturb_optim(
                    self.wrapped_fcn, x0, self.variables, self.config.max_perturbation
                )
                if len(best_soln_value) == 0:
                    best_soln_value.append(x0_val)
                else:
                    best_soln_value.append(min(min(best_soln_value), x0_val))
                if gen >= 2 and np.allclose(
                    best_soln_value[-1], best_soln_value[-2], atol=1e-2, rtol=1e-2
                ):
                    stop_reason = "no_improvement"
                    break

            return OptimizerResult(
                solution_score=best_soln_value[-1],
                solution_history=np.array(best_soln_value),
                solution_vector=x0,
                stop_reason=stop_reason,
            )
        else:
            # ``get``/``set`` are index-addressable-storage operations that only
            # the scalar deck has; a CVTArchive stores by cell.
            if not isinstance(self.soln_deck, SolutionDeck):
                raise NotImplementedError(
                    "optimize_whole_solution_deck requires the scalar SolutionDeck; "
                    f"got {type(self.soln_deck).__name__}."
                )
            # Fill the deck. Without this the loop below read straight out of
            # the np.empty the deck was constructed with -- uninitialised memory
            # as both the starting vector and the incumbent score.
            self.initialize_deck(preserve_percent)
            if len(self.soln_deck) == 0:
                raise ValueError(
                    "optimize_whole_solution_deck needs a non-empty solution deck."
                )
            best_soln_vector: AF | None = None
            # ``len``, not ``archive_size``: dedup/truncate can leave the deck
            # shorter than its configured cap, and ``get`` indexes the array.
            for soln_idx in tqdm(
                range(len(self.soln_deck)), desc="Solution Deck Entry"
            ):
                # .copy(): ``get`` hands back a view into the archive and
                # ``local_perturb_optim`` writes through its argument.
                deck_vector, _, _ = self.soln_deck.get(soln_idx)
                x0 = deck_vector.copy()
                # Tracking the running best as ``[-1]`` rather than
                # ``min(whole_list)`` is equivalent (the list is monotone
                # non-increasing) and O(1) per generation.
                # Not seeded from the deck's stored score: the running best has
                # to correspond to a vector this loop actually produced, or the
                # returned score describes a different point than the returned
                # vector.
                cur_best_soln_value: list[float] = []
                stop_reason = "max_iterations"
                for gen in tqdm(
                    range(self.config.num_generations),
                    desc="Stepwise optimization generations",
                ):
                    x0, x0_val = local_perturb_optim(
                        self.wrapped_fcn,
                        x0,
                        self.variables,
                        self.config.max_perturbation,
                    )
                    if len(cur_best_soln_value) == 0:
                        cur_best_soln_value.append(x0_val)
                    else:
                        cur_best_soln_value.append(min(cur_best_soln_value[-1], x0_val))
                    if gen >= 2 and np.allclose(
                        cur_best_soln_value[-1],
                        cur_best_soln_value[-2],
                        atol=1e-2,
                        rtol=1e-2,
                    ):
                        stop_reason = "no_improvement"
                        break
                self.soln_deck.set(
                    soln_idx, x0, x0_val, stop_reason == "no_improvement"
                )
                # Emptiness first: written the other way round, the ``[-1]`` ran
                # on an empty list before the guard could stop it.
                if (
                    len(best_soln_value) == 0
                    or cur_best_soln_value[-1] < best_soln_value[-1]
                ):
                    best_soln_vector = x0
                    best_soln_value.append(cur_best_soln_value[-1])

            assert best_soln_vector is not None  # the deck is non-empty
            return OptimizerResult(
                solution_score=best_soln_value[-1],
                solution_history=np.array(best_soln_value),
                solution_vector=best_soln_vector,
                stop_reason="none",  # TODO - Is there a better method here?
            )
