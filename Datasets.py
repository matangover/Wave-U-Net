import glob
import os.path
import random
import multiprocessing

import Utils

import numpy as np
import os
import tensorflow as tf
from pathlib import Path

from typing import Dict, Any

def take_random_snippets(sample, keys, input_shape, num_samples):
    # Take a sample (collection of audio files) and extract snippets from it at a number of random positions
    start_pos = tf.random_uniform([num_samples], 0, maxval=sample["length"] - input_shape[0], dtype=tf.int64)
    return take_snippets_at_pos(sample, keys, start_pos, input_shape, num_samples)

def take_all_snippets(sample, keys, input_shape, output_shape):
    # Take a sample and extract snippets from the audio signals, using a hop size equal to the output size of the network
    start_pos = tf.range(0, sample["length"] - input_shape[0], delta=output_shape[0], dtype=tf.int64)
    num_samples = start_pos.shape[0]
    return take_snippets_at_pos(sample, keys, start_pos, input_shape, num_samples)

def take_snippets_at_pos(sample, keys, start_pos, input_shape, num_samples):
    # Take a sample and extract snippets from the audio signals at the given start positions with the given number of samples width
    batch = dict()
    for key in keys:
        signal = sample[key]
        if key.endswith('_score'):
            dtype = tf.uint8
            num_channels = 1
            signal = tf.expand_dims(signal, axis=-1)
        else:
            num_channels = input_shape[1]
            dtype = tf.float32
        
        batch[key] = tf.map_fn(lambda pos: signal[pos:pos + input_shape[0], :], start_pos, dtype=dtype)
        batch[key].set_shape([num_samples, input_shape[0], num_channels])

    return tf.data.Dataset.from_tensor_slices(batch)

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_records(sample_list, model_config, input_shape, output_shape, records_path):
    # Writes samples in the given list as TFrecords into a given path, using the current model config and in/output shapes

    # Compute padding
    if (input_shape[1] - output_shape[1]) % 2 != 0:
        print("WARNING: Required number of padding of " + str(input_shape[1] - output_shape[1]) + " is uneven!")
    pad_frames = (input_shape[1] - output_shape[1]) // 2

    # Set up writers
    num_writers = 1
    writers = [tf.python_io.TFRecordWriter(records_path + str(i) + ".tfrecords") for i in range(num_writers)]

    # Go through songs and write them to TFRecords
    for sample in sample_list:
        try:
            example_proto = process_sample(sample, pad_frames, model_config)
            writers[np.random.randint(0, num_writers)].write(example_proto.SerializeToString())
        except (AssertionError, ValueError) as e:
            print('Error (%s):\n%s' % (multiprocessing.current_process().name, e))

    for writer in writers:
        writer.close()

def parse_record(example_proto, shape, model_config):
    # Parse record from TFRecord file
    if model_config['sources_to_mix'] is None:
        tracks_to_read = model_config['source_names'] + ['mix']
    else:
        tracks_to_read = model_config['sources_to_mix']

    features = {
        key: tf.FixedLenSequenceFeature([], allow_missing=True, dtype=tf.float32)
        for key in tracks_to_read
    } # type: Dict[str, Any]
    features["length"] = tf.FixedLenFeature([], tf.int64)
    features["channels"] = tf.FixedLenFeature([], tf.int64)
    if model_config['score_informed']:
        features.update({
            (s + '_score'): tf.FixedLenFeature([], tf.string)
            for s in model_config['source_names']
        })

    parsed_features = tf.parse_single_example(example_proto, features)

    if model_config['sources_to_mix'] is not None:
        mix_sources(model_config['sources_to_mix'], parsed_features)

    # Reshape
    length = tf.cast(parsed_features["length"], tf.int64)
    channels = tf.constant(shape[-1], tf.int64) #tf.cast(parsed_features["channels"], tf.int64)
    sample = dict()
    for key in model_config['source_names'] + ['mix']:
        sample[key] = tf.reshape(parsed_features[key], tf.stack([length, channels]))
    sample["length"] = length
    sample["channels"] = channels

    if model_config['score_informed']:
        for key in model_config['source_names']:
            score = tf.io.decode_raw(parsed_features[key + '_score'], tf.uint8)
            # This hack is needed because chorales_synth_v6 was generated wrongly --
            # The audio was zero-padded (due to context=True) but the scores were
            # not zero-padded.
            if model_config['data_path'].endswith('chorales_synth_v6'):
                input_shape, output_shape = Utils.get_separator_shapes(model_config)
                pad_frames = (input_shape[1] - output_shape[1]) // 2
                score = tf.pad(score, [[pad_frames, pad_frames]], mode="constant", constant_values=0.0)
            sample[key + '_score'] = score

    return sample

def preprocess_dataset(model_config, input_shape, output_shape, tiny=False):
    print("Reading chorales")
    chorales = getSynthesizedChorales(model_config["chorales_path"], model_config["score_informed"])
    if tiny:
        dataset = {
            "train": chorales[:1],
            "valid": chorales[1:2],
            "test": chorales[2:3]
        }
    else:
        # Total chorales: 371. After removal of chorales with more than 4 staves: 351 chorales.
        dataset = {
            "train": chorales[:270],
            "valid": chorales[270:320],
            "test": chorales[320:]
        }
    # Convert audio files into TFRecords now

    # The dataset structure is a dictionary with "train", "valid", "test" keys, whose entries are lists, where each element represents a song.
    # Each song is represented as a dictionary containing elements mix, acc, vocal or mix, bass, drums, other, vocal depending on the task.

    num_cores = multiprocessing.cpu_count()

    for curr_partition in ["train", "valid", "test"]:
        print("Writing " + curr_partition + " partition...")

        # Shuffle sample order
        sample_list = dataset[curr_partition]
        random.shuffle(sample_list)

        # Create folder
        partition_folder = os.path.join(model_config["data_path"], curr_partition)
        os.makedirs(partition_folder)

        part_entries = int(np.ceil(float(len(sample_list) / float(num_cores))))
        processes = list()
        for core in range(num_cores):
            train_filename = os.path.join(partition_folder, str(core) + "_")  # address to save the TFRecords file
            sample_list_subset = sample_list[core * part_entries:min((core + 1) * part_entries, len(sample_list))]
            proc = multiprocessing.Process(target=write_records,
                            args=(sample_list_subset, model_config, input_shape, output_shape, train_filename))
            proc.start()
            processes.append(proc)
        for p in processes:
            p.join()

    print("Dataset ready!")

def get_dataset(model_config, input_shape, output_shape, partition):
    main_folder = model_config["data_path"]
    # Finally, load TFRecords dataset based on the desired partition
    dataset_folder = os.path.join(main_folder, partition)
    records_files = glob.glob(os.path.join(dataset_folder, "*.tfrecords"))
    if model_config["shuffle_dataset"]:
        random.shuffle(records_files)
    dataset = tf.data.TFRecordDataset(records_files)
    dataset = dataset.map(
        lambda x: parse_record(x, input_shape[1:], model_config),
        num_parallel_calls=model_config["num_workers"])
    dataset = dataset.prefetch(10)

    # Take random samples from each song
    keys = model_config["source_names"] + ["mix"]
    if model_config["score_informed"]:
        keys += [source + "_score" for source in model_config["source_names"]]
    
    if partition == "train":
        dataset = dataset.flat_map(lambda x: take_random_snippets(x, keys, input_shape[1:], model_config["num_snippets_per_track"]))
    else:
        dataset = dataset.flat_map(lambda x: take_all_snippets(x, keys, input_shape[1:], output_shape[1:]))
    dataset = dataset.prefetch(100)

    if partition == "train" and model_config["augmentation"]: # If its the train partition, activate data augmentation if desired
            dataset = dataset.map(Utils.random_amplify, num_parallel_calls=model_config["num_workers"]).prefetch(100)

    if model_config["multiple_source_training"]:
        dataset = dataset.flat_map(lambda sample: split_sources(sample, model_config))
    # Cut source outputs to centre part
    dataset = dataset.map(lambda x: Utils.crop_sample(x, (input_shape[1] - output_shape[1])//2)).prefetch(100)

    if partition == "train": # Repeat endlessly and shuffle when training
        dataset = dataset.repeat()
        if model_config["shuffle_dataset"]:
            dataset = dataset.shuffle(buffer_size=model_config["cache_size"])

    dataset = dataset.batch(model_config["batch_size"], drop_remainder=True)
    dataset = dataset.prefetch(1)

    return dataset


def getSynthesizedChorales(chorale_dir, score_informed):
    chorale_dir = Path(chorale_dir)
    mixes = sorted((chorale_dir / 'mix').glob('chorale_*_mix.wav'))
    samples = []
    voices = ['soprano', 'alto', 'tenor', 'bass']
    for mix in mixes:
        chorale_name = mix.name
        sample = {"mix" : str(mix)}
        for voice in voices:
            sample[voice] = str(chorale_dir / 'audio_mono' / chorale_name.replace('mix', voice))
        if score_informed:
            for voice in voices:
                sample[voice + '_score'] = str((chorale_dir / 'midi' / chorale_name.replace('mix', voice)).with_suffix('.mid'))
        samples.append(sample)

    return samples

def read_score(score_filename, length, sample_rate):
    """
    Read MIDI file into array of 'currently active pitch' or 0.
    Supports only monophonic MIDI files (one note at a time).
    """
    import mido
    with mido.MidiFile(score_filename) as score_midi:
        score = np.zeros(length, dtype=np.uint8)

        current_time = 0
        on_since = {} # type: Dict[int, float]
        for msg in score_midi:
            current_time += msg.time
            if msg.is_meta:
                continue
            if msg.type == 'note_on' and msg.velocity != 0:
                if on_since:
                    raise ValueError('Error parsing midi (%s): note_on (%s) while another note is active at time %s' % (score_filename, msg.note, current_time))
                on_since[msg.note] = current_time
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note not in on_since:
                    raise ValueError('Error parsing midi (%s): note_off without note_on: %s at time %s' % (score_filename, msg.note, current_time))
                note_on_since = on_since.pop(msg.note)
                start_sample, end_sample = time_to_sample(note_on_since, sample_rate), time_to_sample(current_time, sample_rate)
                score[start_sample:end_sample] = msg.note
        
        return score

def time_to_sample(time, sample_rate):
    return int(round(time * sample_rate))

def process_sample(sample, pad_frames, model_config):
    print("Reading song: %s (process: %s)" % (os.path.basename(sample["mix"]), multiprocessing.current_process().name))
    audio_tracks = dict()
    all_keys = model_config["source_names"] + ["mix"]

    for key in all_keys:
        audio, _ = Utils.load(sample[key], sr=model_config["expected_sr"], mono=model_config["mono_downmix"])

        if not model_config["mono_downmix"] and audio.shape[1] == 1:
            print("WARNING: Had to duplicate mono track to generate stereo")
            audio = np.tile(audio, [1, 2])

        audio_tracks[key] = audio

    # Pad at beginning and end with zeros
    audio_tracks = {key: np.pad(audio_tracks[key], [(pad_frames, pad_frames), (0, 0)], mode="constant", constant_values=0.0) for key in audio_tracks.keys()}

    # All audio tracks must be exactly same length and channels
    length = audio_tracks["mix"].shape[0]
    channels = audio_tracks["mix"].shape[1]
    for key, audio in audio_tracks.items():
        assert audio.shape[0] == length, 'Length of track (%s) not equal to length of mix (%s): %s' % (
            audio.shape[0], length, sample[key])
        assert audio.shape[1] == channels, 'Channels of track (%s) not equal to channels of mix (%s): %s' % (
            audio.shape[1], channels, sample[key])

    # Write to TFrecords the flattened version
    feature = {key: _floats_feature(audio_tracks[key]) for key in all_keys}
    feature["length"] = _int64_feature(length)
    feature["channels"] = _int64_feature(channels)
    if model_config["score_informed"]:
        for source in model_config["source_names"]:
            # TODO: bug! should pad score with pad_frames.
            raise RuntimeError("Fix this bug before you run this script again, and remove hack code .")
            feature[source + '_score'] = _bytes_feature(
                read_score(sample[source + '_score'], length, model_config['expected_sr']).tobytes())

    return tf.train.Example(features=tf.train.Features(feature=feature))

def split_sources(sample, model_config):
    dataset = None
    for source in model_config["source_names"]:
        new_sample = {}
        new_sample["mix"] = sample["mix"]
        new_sample["source"] = sample[source]
        if model_config["score_informed"]:
            new_sample["source_score"] = sample[source + "_score"]
        new_dataset = tf.data.Dataset.from_tensors(new_sample)
        if dataset is None:
            dataset = new_dataset
        else:
            dataset = dataset.concatenate(new_dataset)

    if model_config["add_random_score_samples"]:
        random_score_sample = get_random_score_sample(sample, model_config)
        assert dataset is not None
        dataset = dataset.concatenate(tf.data.Dataset.from_tensors(random_score_sample))

    return dataset

def get_random_score_sample(sample, model_config):
    # NOTE: This works only when source_names includes two or more sources.
    # Take the 'average' between two neighboring random source. E.g. average between tenor and bass.
    concatenated_scores = tf.concat([sample[source + '_score'] for source in model_config["source_names"]], 1)
    source_index = tf.random.uniform([], 0, len(model_config["source_names"]) - 1, dtype=tf.int32)
    sources_to_average = concatenated_scores[:, source_index:source_index+2]
    average_score = tf.math.reduce_mean(sources_to_average, 1)
    # Keep zeros where any of the averaged sources was zero.
    zero_score = tf.zeros_like(average_score)
    for source_score_index in (0, 1):
        source_score = sources_to_average[:, source_score_index]
        average_score = tf.where(tf.equal(source_score, 0), zero_score, average_score)

    return {
        "mix": sample["mix"],
        "source": tf.zeros_like(sample[model_config["source_names"][0]]),
        "source_score": tf.expand_dims(average_score, 1)
    }


def mix_sources(sources_to_mix, sample):
    mix = tf.add_n([sample[s] for s in sources_to_mix])
    # If the mixing caused clipping, scale mix and sources accordingly.
    max_abs = tf.reduce_max(tf.abs(mix))
    scaling_factor = tf.minimum(1.0 / max_abs, 1)
    sample['mix'] = mix * scaling_factor
    for source in sources_to_mix:
        sample[source] = sample[source] * scaling_factor