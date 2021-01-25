
#from collections import namedtuple

#SAP = namedtuple("SAP", ["state", "action"])
from typing import NamedTuple

class SAP(NamedTuple):
    
    state: int
    action: str
    
    def __str__(self):
        return f"{self.state} \n {self.action}"
