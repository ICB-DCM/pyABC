import scipy as sp


# models which only generate data, summary stats independent

def model1(x, y, z):
    return sp.rand() * x * y + z


def model2(par):
    return sp.rand() * par["x"] * par["y"] + par["z"]


# def models which accept, reject

## exception for early rejection? better not!

class Reject(Exception):
    pass


def model3(x, y, z):
    k = 0
    for k in range(z):
        if k > y:
            raise Reject()
    return k * x * sp.rand()


def model4(x, y, z):
    k = 0
    for k in range(z):
        if k > y:
            return
    return x * y


class Result:
    pass


class Accepted(Result):
    pass


class Rejected(Result):
    pass


def model5(x, y, z):
    k = 0
    for k in range(z):
        if k > y:
            return Rejected(k)
    return x * y


def model6(x, y, z):
    k = 0
    for k in range(z):
        if k > y:
            return False, k
    return x * y

