from .minni import init
from .minni import Metric

# Automatically initialize when importing the module
if not init.done:
    init.done = True
    init()
