
import tensorflow_datasets as tfds


def load_tfds(dataset="glue/sst2", do_sort=True, **kw):
  """Load from TFDS, with optional sorting."""
  # Materialize to NumPy arrays.
  # This also ensures compatibility with TF1.x non-eager mode, which doesn't
  # support direct iteration over a tf.data.Dataset.
  ret = list(tfds.as_numpy(tfds.load(dataset, download=True, try_gcs=True, **kw)))
  print(ret)
  if do_sort:
    # Recover original order, as if you loaded from a TSV file.
    ret.sort(key=lambda ex: ex['idx'])
  return ret