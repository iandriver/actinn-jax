# -*- coding: utf-8 -*-

__author__ = 'Ian Driver'
__email__ = 'driver.ian@gmail.com'
__version__ = '0.3.0'


from . import actinn_utils
from . import actinn_predict
from . import hierarchy
from .actinn_predict import (
    ReferenceModel,
    train_reference,
    predict,
    celltype_predict_actinn,
)
from .hierarchy import (
    HierarchicalReferenceModel,
    discover_hierarchy,
    build_hierarchical_reference,
    annotate,
    bundled_reference,
    RefinedReference,
    detect_present_classes,
    refine_to_query,
    refine_to_tissue,
)

__all__ = [
    "actinn_utils",
    "actinn_predict",
    "hierarchy",
    "ReferenceModel",
    "train_reference",
    "predict",
    "celltype_predict_actinn",
    # two-stage workflow
    "HierarchicalReferenceModel",
    "discover_hierarchy",
    "build_hierarchical_reference",
    "annotate",
    "bundled_reference",
    # query-adaptive refinement
    "RefinedReference",
    "detect_present_classes",
    "refine_to_query",
    "refine_to_tissue",
]
