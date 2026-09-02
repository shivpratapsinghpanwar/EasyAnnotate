"""Annotation formats: a neutral model, plus readers and writers around it."""

from .model import Annotation, AnnotationError, Point, Shape

__all__ = ["Annotation", "AnnotationError", "Point", "Shape"]
