from sacred import Experiment
from Config import config_ingredient
import tensorflow as tf
import Datasets
from pathlib import Path
import Utils
import librosa
import os
import tempfile
import itertools
import numpy as np
from Models.UnetAudioSeparator import get_score

tf.enable_eager_execution()

ex = Experiment('Wave-U-Net', ingredients=[config_ingredient])


def extract_partition(partition, model_config):
    output_dir = tempfile.mkdtemp()
    records_dir = Path(model_config['data_path']) / partition
    records_paths = sorted(records_dir.glob('*.tfrecords'))
    records_files = map(str, records_paths)
    print('Partition: ' + partition)
    print('Parsing record files:\n\t' + '\n\t'.join(records_files))
    dataset = tf.data.TFRecordDataset(records_files)
    input_shape, _output_shape = Utils.get_separator_shapes(model_config)
    dataset = dataset.map(
        lambda x: Datasets.parse_record(
            x, input_shape[1:], model_config))
    
    dataset = dataset.take(10)
    for song_index, song in enumerate(dataset):
        print('\tSong %s: %s samples' % (song_index + 1, int(song['length'])))
        os.makedirs('%s/%s' % (output_dir, song_index))
        for source in model_config['source_names'] + ['mix']:
            librosa.output.write_wav(
                '%s/%s/%s.wav' % (output_dir, song_index, source),
                tf.squeeze(song[source]).numpy(),
                model_config['expected_sr']
            )


def extract_batches(partition, model_config):
    output_dir = tempfile.mkdtemp()
    print(f'Output to {output_dir}')
    input_shape, _output_shape = Utils.get_separator_shapes(model_config)
    dataset = Datasets.get_dataset(model_config, input_shape, _output_shape, partition)
    dataset = dataset.take(2)
    dataset = dataset.apply(tf.data.experimental.unbatch())
    for snippet_index, snippet in enumerate(dataset):
        print('\tSnippet %s' % (snippet_index + 1))
        snippet_dir = Path(output_dir) / str(snippet_index)
        os.makedirs(snippet_dir)
        write_audio('mix', model_config, snippet_dir, snippet)
        for source in model_config['separator_source_names']:
            write_audio(source, model_config, snippet_dir, snippet)
            if model_config['score_informed']:
                write_score(source, model_config, snippet_dir, snippet)

def write_audio(source_name, model_config, output_dir, sample):
    librosa.output.write_wav(
        str(output_dir / source_name) + '.wav',
        tf.squeeze(sample[source_name]).numpy(),
        model_config['expected_sr']
    )

def write_score(source_name, model_config, output_dir, sample):
    score = tf.squeeze(sample[source_name + '_score'], -1).numpy()
            
    with open(str(output_dir / source_name) + '_score.txt', 'w') as output:
        output.write('\n'.join(f'{midi_to_note(value)} - {len(list(group)):6}' for value, group in itertools.groupby(score)))
    
    processed_score = get_score(one_item_batch(score), model_config['score_type'], model_config)
    save_processed_score(str(output_dir / source_name) + '_score_processed', processed_score, model_config)

    input_shape, output_shape = Utils.get_separator_shapes(model_config)
    crop_frames = (input_shape[1] - output_shape[1])//2
    score_cropped = score[crop_frames:-crop_frames]
    with open(str(output_dir / source_name) + '_score_cropped.txt', 'w') as output:
        output.write('\n'.join(f'{midi_to_note(value)} - {len(list(group)):6}' for value, group in itertools.groupby(score_cropped)))

    processed_score_cropped = get_score(one_item_batch(score_cropped), model_config['score_type'], model_config)
    save_processed_score(str(output_dir / source_name) + '_score_cropped_processed', processed_score_cropped, model_config)

def midi_to_note(midi):
    return 0 if midi == 0 else librosa.midi_to_note(midi)

def save_processed_score(name, score, model_config):
    score = tf.squeeze(score, (0, 2))
    if model_config['score_type'] == 'pure_tone_synth':
        librosa.output.write_wav(name + '.wav', score.numpy(), model_config['expected_sr'])
    else:
        np.savetxt(name + '.txt', score, fmt='%.3f', delimiter=', ')

def one_item_batch(score):
    return tf.expand_dims(tf.expand_dims(score, 0), 2)

@ex.automain
def run(cfg):
    cfg['model_config']['shuffle_dataset'] = False
    extract_batches('train', cfg['model_config'])