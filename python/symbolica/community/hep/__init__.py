"""High-energy physics tools from FeynKit.

Models, Feynman diagrams, generation, CFFs, tensor reduction, and kinematics
are available directly in this namespace, for example ``hep.FeynmanDiagram``.
Native builds also provide one-loop integrals through ``hep.oneloop`` and exact
parametric and Laporta IBP solving through ``hep.IBPFamily``.
"""

from ..hep_native import *

initialize_module()
del initialize_module
