from enum import Enum


class EnumType(Enum):
    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, EnumType):
            return self.value == other.value
        return self.value == other

    def __hash__(self):
        return hash(self.value)

    def __contains__(self, item):
        return item in self.value

    def __iter__(self):
        return iter(self.value)
