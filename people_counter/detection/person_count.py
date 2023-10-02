# The smaller this value, the harder it is for the smoothed value to change
SMOOTH_FACTOR = 0.05

class PersonCount:
    def __init__(self, initial_value: int):
        self._current_count = float(initial_value)

    def update(self, new: int):
        next_count = smooth_value(new, self._current_count)
        has_value_changed = round(next_count) != self.current
        self._current_count = next_count

        return has_value_changed

    @property
    def current(self) -> int:
        return round(self._current_count)


def smooth_value(new, previous) -> float:
    return SMOOTH_FACTOR * new + (1 - SMOOTH_FACTOR) * previous
