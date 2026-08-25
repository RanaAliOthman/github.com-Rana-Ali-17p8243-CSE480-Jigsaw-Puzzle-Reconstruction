"""Reusable, independently testable classical jigsaw vision library.

The pipeline is ``frame_extraction`` -> ``piece_geometry`` ->
``edge_compatibility`` -> ``solver``, built on the from-scratch primitives in
``enhancement``, ``thresholding``, ``edge_detection``, ``segmentation`` and
``contour_extraction``.
"""
from .piece_geometry import CanonicalPiece, Side, SideType

__all__ = ["CanonicalPiece", "Side", "SideType"]
