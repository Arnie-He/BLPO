from enum import Enum

class DynParam(Enum):
    """An enum class for dynamic parameters determined by the environment"""
    ActionCount = 0

def init(env, env_params, model_params):
    """Initializes dynamic model parameters given an environment."""
    for p in range(len(model_params)):
        if model_params[p] == DynParam.ActionCount:
            model_params[p] = env.action_space(env_params).n
    return model_params