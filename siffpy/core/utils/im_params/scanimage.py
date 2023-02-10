from functools import reduce
from operator import add
import re

def _unsafe_eval(val):
    """
    NOT ACTUALLY TOTALLY SAFE just won't crash the interpreter
    on standard ScanImage output!!!!!!!
    """
    try:
        ret = eval(val)
    except NameError:
        if isinstance(val, str):
            try:
                ret = eval(val.capitalize())
            except:
                ret = val
        else:
            ret = val
    except:
        ret = val
    
    return ret
            
class ScanImageModule():
    """ Generic module for each ScanImage module stored in header data """
    
    def __init__(self, name : str):
        self.module_name = name
        self.submodules : dict [str, ScanImageSubModule]= {}

    def add_param(self, key : str, val : str):
        split_key = key.split('.')
        if len(split_key) == 1:
            setattr(self, key, _unsafe_eval(val))
        else:
            module_name = split_key[0]
            if re.match(r"h[A-Z].*", module_name):
                module_name = module_name[1:]
            if not hasattr(self, module_name):
                self.submodules[module_name] = ScanImageSubModule(module_name)
            if len(split_key) > 2:
                parname = '.'.join(split_key[1:])
            else:
                parname = split_key[1]
            self.submodules[module_name].add_param(parname, val)

    def __getitem__(self, key : str):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            super().__getitem__(key)

    def __getattr__(self, __name : str):
        if __name in self.submodules:
            return self.submodules[__name]
        else:
            return super().__getattribute__(__name)
        
    def pretty_dict(self)->str:
        return reduce(
            add,
            (
                f"\n\t{str(k)} : {str(v)}" for k, v in self.__dict__.items()
                if not k == 'module_name'
            )
        )

    def __str__(self):
        return f"{self.module_name} module: {self.pretty_dict()}"
    
    def __repr__(self):
        return self.__str__()

class ScanImageSubModule(ScanImageModule):
    """
    For submodules of ScanImage modules
    """
    pass

class Scanfield():
    """ Generic ScanImage scanfield """
    def __init__(self, scanfield_dict : dict):
        for key, val in scanfield_dict.items():
            setattr(self, key, _unsafe_eval(val))

class SIROI():
    """
    ScanImage ROIs -- NOT to be confused with siffpy ROIs
    """
    def __init__(self, roi_dict : dict):
        for key, val in roi_dict.items():
            if key == 'scanfields':
                if not isinstance(val, dict):
                    raise NotImplementedError(
                        """
                        Scanfields is not simply a dictionary,
                        meaning that you're probably using
                        mROI functionality. Yay! Send
                        the code to Stephen so he can
                        implement it.
                        """
                    )
                self.scanfields = Scanfield(val)
            else:
                setattr(self, key, _unsafe_eval(val))

class ROIGroup():
    """ Generic ROI group for ScanImage ROI groups """
    def __init__(self, roi_dict : dict):
        for key, val in roi_dict.items():
            if key == 'rois':
                if not isinstance(val,dict):
                    raise NotImplementedError(
                        """
                        ROIs is not simply a dictionary,
                        meaning that you're probably using
                        mROI functionality. Yay! Send
                        the code to Stephen so he can
                        implement it.
                        """
                    )
                self.rois = SIROI(val)
            else:    
                setattr(self, key, _unsafe_eval(val))