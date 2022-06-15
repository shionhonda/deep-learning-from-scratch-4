State = tuple[int, int]
Value = dict[State, float]
Action = int
Q = dict[tuple[State, Action], float]
Policy = dict[State, dict[Action, float]]
Reward = float
