

class MapWrapperDefault():
    """
        This is the basic map_wrapper implementation required for code compatibility reasons.
        There should be no need do initialize this on the user-level.
    """
    def __init__(self,map_fun=map):
        self.map_fun = map_fun;

    def set_map_fun(self,map_fun):
        self.map_fun = map_fun;

    def wrap_map(self,lambdafun, args):
        # sanity checks
        return self.map_fun(lambdafun, args)




class MapWrapperDistribDemo(map):
    """
        This implementation of the map wrapper demonstrates the idea behind the map_wrapper object.
    """
    def __init__(self, map_fun=map, test_string=""):
        self.map_fun = map_fun;
        self.test_string = test_string


    def set_map_fun(self, map_fun):
        self.map_fun = map_fun;


    def wrap_map(self, lambdafun, args):
        # sanity checks
        print(self.test_string)
        print("If I was a real distributed mapper, I'd now start an army of workers...")
        print("They'd do my bidding... I'd leave orders for them in a DB, they'd gather them and return their results...")
        print("Once I'd have decided that my results are sufficient, I'd return them as a list")
        print("However, I'm a poor imitation and still only do a simple map() call and return the result")
        print("Now I'm depressed. Are you happy?")
        result = self.map_fun(lambdafun, args)
        return result
