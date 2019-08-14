from sacred import Experiment
from Config import config_ingredient
import tensorflow as tf
import Datasets
from pathlib import Path
import Utils
from typing import Iterable

tf.enable_eager_execution()

ex = Experiment('Wave-U-Net', ingredients=[config_ingredient])

def check_partition(partition, model_config):
    records_dir = Path(model_config['data_path']) / partition
    records_files = sorted(records_dir.glob('*.tfrecords')) # type: Iterable[str]
    records_files = map(str, records_files)
    print('Partition: ' + partition)
    print('Parsing record files:\n\t' + '\n\t'.join(records_files))
    dataset = tf.data.TFRecordDataset(records_files)
    input_shape, _output_shape = Utils.get_separator_shapes(model_config)
    dataset = dataset.map(
        lambda x: Datasets.parse_record(
            x, input_shape[1:], model_config))
    
    for i, song in enumerate(dataset):
       print('\tSong %s: %s samples' % (i + 1, int(song['length'])))
    print('Total in %s partition: %s songs' % (partition, i + 1))

@ex.automain
def run(cfg):
    for partition in ['train', 'valid', 'test']:
        check_partition(partition, cfg['model_config'])