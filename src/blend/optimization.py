import random
import types


def change_optimization_order(*seekers):
    """Changes the optimization order of seekers.

    Modifies the cost method of each seeker to return its index.

    Args:
        *seekers: Variable length argument list of seekers.
    """
    for i, seeker in enumerate(seekers):
        seeker.cost = types.MethodType(lambda self, v=i: v, seeker)


def give_same_type(*seekers):
    """Assigns the same random cost to all seekers.

    Args:
        *seekers: Variable length argument list of seekers.
    """
    rnd = random.randint(0, 100000)
    for seeker in seekers:
        seeker.cost = types.MethodType(lambda self: rnd, seeker)
