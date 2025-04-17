import math


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor

def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        print(f"WARNING --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}")
        return new_size