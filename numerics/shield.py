"""Lane A, task A5: shield"""
from contracts.types import Action, Obs, Params, ProgramFn


def shielded_action(recovered: ProgramFn, rec_params: Params, fixed: ProgramFn, fix_params: Params,
                    obs: Obs, bb_action: Action) -> Action:
    """if recovered(obs) != fixed(obs): return fixed(obs), else return bb_action"""
    raise NotImplementedError
