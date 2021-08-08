""" Parser factory

Copyright 2020 Ross Wightman
"""
from .parser_coco import CocoParser
from .parser_voc import VocParser
from .parser_open_images import OpenImagesParser
from .parser_xray import XrayParser

def create_parser(name, **kwargs):
    if name == 'coco':
        parser = CocoParser(**kwargs)
    elif name == 'voc':
        parser = VocParser(**kwargs)
    elif name == 'openimages':
        parser = OpenImagesParser(**kwargs)
    elif name == 'xray':
        parser = XrayParser(**kwargs)
    elif name == 'rsnapneu':
        parser = XrayParser(**kwargs)
    elif name == 'warmup':
        parser = XrayParser(**kwargs)
    else:
        assert False, f'Unknown dataset parser ({name})'
    return parser
