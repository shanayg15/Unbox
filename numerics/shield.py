"""Lane A, task A5: shield"""
from contracts.types import Action, Obs, Params, ProgramFn


def shielded_action(recovered: ProgramFn, rec_params: Params, fixed: ProgramFn, fix_params: Params,
                    obs: Obs, bb_action: Action) -> Action:
    """if recovered(obs) != fixed(obs): return fixed(obs), else return bb_action"""
    try:
        recovered_action = recovered(obs, rec_params)
        fixed_action = fixed(obs, fix_params)
    except Exception:       # a crashing program can't vouch for an override, so the black box acts
        return bb_action
    return fixed_action if recovered_action != fixed_action else bb_action
