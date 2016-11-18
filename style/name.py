import warnings
from collections import UserDict

name = {
    "CDF": "CDF",
    "PDF": "PDF",
    "model": "Model",
}



class Tex:
    def __init__(self, *args):
        if len(args) == 0:
            self.text = ""
        elif len(args) == 1:
            self.text = str(args[0]).strip("$")
        else:
            self.text = sum(map(type(self), args), type(self)()).text

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "$" + self.text + "$"

    def __add__(self, other):
        try:
            text = other.text
        except AttributeError:
            text = other
        return Tex(self.text + text)

    def expectation(self):
        return expectation(self)

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item.start == 1 and item.stop == -1:
                return self.text[1:-1]
        raise Exception("Slicing only defined for 1:-1")


def expectation(tex: Tex):
    tex = Tex("\mathbb{E} \left [", tex, "\\right]")
    return tex


class NameDict(UserDict):
    def __getitem__(self, item):
        return super().__getitem__(item)[1]

    def __missing__(self, item):
        warnings.warn("Missing key: {}".format(item))
        return "!!!{}!!!".format(item)

    def full_name(self, item):
        full_name = super().__getitem__(item)[0]
        return full_name


names = {
    "correlation": ("", "$c$"),
    "time": ("", "$t$"),
    "weight": ("", "$w$"),
    "pair_correlation": ("", "$c$"),
    "cosine_similarity": ("", "$c_{sim}$")
}


latex_name = NameDict(names)