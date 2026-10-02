"""Tile library: the candidate photos a mosaic is built from.

`library` keeps an on-disk cache of every tile image (metadata plus a small
analysis thumbnail), `ingest` fills it, `crops` chooses the windows of each
tile that fit a region shape, and `descriptors` turns those windows into the
perceptual vectors that matching compares.
"""
