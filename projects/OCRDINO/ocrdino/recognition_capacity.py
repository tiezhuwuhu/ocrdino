"""Geometry-only capacity rules shared by model and offline data audits."""
import math
from numbers import Real

DEFAULT_BUCKETS = (32, 64, 128, 256, 500)
DEFAULT_WIDTH_SCALE = 16.0


def validate_buckets(buckets, name='buckets'):
    values = tuple(buckets)
    if (not values or any(isinstance(value, bool) or not isinstance(value, int)
                          or value <= 0 for value in values)
            or list(values) != sorted(set(values))):
        raise ValueError(name + ' must be increasing, unique positive integers')
    return values


def validate_width_scale(scale, name='width_scale'):
    if (isinstance(scale, bool) or not isinstance(scale, Real)
            or not math.isfinite(scale) or scale <= 0):
        raise ValueError(name + ' must be positive and finite')
    return float(scale)


def select_geometry_bucket(width, height, buckets=DEFAULT_BUCKETS,
                           width_scale=DEFAULT_WIDTH_SCALE):
    """Select the smallest covering bucket, capped by its largest entry.

    This function never accepts text or target length. Runtime tensor code
    uses the same float64 ceil/aspect-ratio rule; resized/clipped coordinates
    must still be checked at training time because geometry may have changed.
    """
    buckets = validate_buckets(buckets)
    scale = validate_width_scale(width_scale)
    if any(isinstance(value, bool) or not isinstance(value, Real)
           or not math.isfinite(value) or value <= 0 for value in (width, height)):
        raise ValueError('Box width and height must be positive and finite')
    desired = float(width) / float(height) * scale
    # Finite inputs can overflow their ratio; overflow means the largest bucket.
    if not math.isfinite(desired) or desired > buckets[-1]:
        return buckets[-1]
    requested = math.ceil(desired)
    return next(bucket for bucket in buckets if bucket >= requested)


def required_output_positions(text):
    """One ordered query per Unicode code point, plus exactly one EOS."""
    if not isinstance(text, str):
        raise ValueError('Transcription must be a string')
    return len(text) + 1