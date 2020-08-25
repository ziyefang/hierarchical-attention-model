
import tensorflow_datasets as tfds

ret = list(tfds.as_numpy(tfds.load("glue/sst2", download=True, try_gcs=True, split="train")))
ret.sort(key=lambda ex: ex['idx'])
print(ret)
LABELS = ['0', '1']
examples = []
for ex in ret:
  examples.append({
    'sentence': ex['sentence'].decode('utf-8'),
    'label': LABELS[ex['label']],
  })
print(examples)
