"""Sheet music -> notes -> audio.

The pipeline is split into stages that mirror the original notebook:
staff detection, notehead segmentation, CNN classification, pitch
estimation, and finally score assembly / export.
"""
