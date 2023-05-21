import copy

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

##############
# cfg_holder #
##############

@singleton
class cfg_unique_holder(object):
    def __init__(self):
        self.cfg = None
        # this is use to track the main codes.
        self.code = set()
    def save_cfg(self, cfg):
        self.cfg = copy.deepcopy(cfg)
    def add_code(self, code):
        """
        A new main code is reached and 
            its name is added.
        """
        self.code.add(code)
