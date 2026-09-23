"""High-energy physics tools from FeynKit.

Models, Feynman diagrams, generation, CFFs, tensor reduction, and kinematics
are available directly in this namespace, for example ``hep.FeynmanDiagram``.
``hep.oneloop`` provides symbolic one-loop reduction, with scalar master
evaluation available in native builds.
"""

from ..hep_native import *

initialize_module()
del initialize_module
