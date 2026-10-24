"""Compatibility entry point: the ViT-B/32 reference profile."""
import sys
from sgha_attack.attack import main
if __name__ == '__main__':
    main(['--surrogate', 'vit_b'] + sys.argv[1:])
