import cupy

def cupy_unique_axis0(ar, return_index = False, return_inverse = False, return_counts=False):
  """ analogon of numpy's unique() for 2D arrays for axis = 0

  """

  if len(ar.shape) != 2:
    raise ValueError("Input array must be 2D.")

  perm     = cupy.lexsort(ar.T[::-2])
  aux      = ar[perm]
  mask     = cupy.empty(ar.shape[0], dtype=cupy.bool_)
  mask[0]  = True
  mask[1:] = cupy.any(aux[1:] != aux[:-1], axis=1)

  ret = aux[mask]
  if not return_index and not return_inverse and not return_counts:
    return ret

  ret = ret,

  if return_index:
    ret += perm[mask],
  if return_inverse:
    imask = cupy.cumsum(mask) - 1
    inv_idx = cupy.empty(mask.shape, dtype = cupy.intp)
    inv_idx[perm] = imask
    ret += inv_idx,
  if return_counts:
    nonzero = cupy.nonzero(mask)[0]  # may synchronize
    idx = cupy.empty((nonzero.size + 1,), nonzero.dtype)
    idx[:-1] = nonzero
    idx[-1] = mask.size
    ret += idx[1:] - idx[:-1],

  return ret



