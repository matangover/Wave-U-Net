import museval
import soundfile
import numpy as np
import librosa
import csv
from typing import Dict, NamedTuple
import pandas
import os
import glob
import argparse
from dotenv import load_dotenv
load_dotenv()

test_chorales = [
    '335', '336', '337', '338', '339', '340', '341', '342', '343', '345', '346',
    '349', '350', '351', '352', '354', '355', '356', '357', '358', '359', '360',
    '361', '363', '364', '365', '366', '367', '369', '370', '371'
]
all_source_names = ['soprano', 'alto', 'tenor', 'bass']
source_abbrev_to_source_name = {
    'S': 'soprano',
    'A': 'alto',
    'T': 'tenor',
    'B': 'bass'
}
# Sample rates of the mixture/parts audio files.
dataset_sample_rates = {
    'chorales_synth': 22050,
}
dataset_base = 'datasets'

def read_sheet(key, sheet_name, **kwargs):
    url = f'https://docs.google.com/spreadsheets/d/{key}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    return pandas.read_csv(url, **kwargs)

def read_models(sheet_id):
    models = read_sheet(sheet_id, 'Models', dtype={'model': str, 'filters_per_layer': 'Int64'}, skip_blank_lines=True)
    models = models[models.model.notnull()]
    models.set_index('model', inplace=True, verify_integrity=True)
    return models

def read_sources(path_template, source_names, sr):
    sources = []
    for source_name in source_names:
        audio_filename = path_template % source_name
        audio, audio_sr = soundfile.read(audio_filename)
        assert audio_sr == sr, f'Expected sample rate {sr} but got {audio_sr} in file {audio_filename}'
        sources.append(audio)
    return np.stack(sources)

def get_snr(reference_sources, estimated_sources, frame_size_samples):
    num_sources, _num_samples = reference_sources.shape
    source_snrs = []
    for i in range(num_sources):
        reference_frames = librosa.util.frame(reference_sources[i, :], frame_size_samples, frame_size_samples)
        estimate_frames = librosa.util.frame(estimated_sources[i, :], frame_size_samples, frame_size_samples)
        signal_energy = np.sum(reference_frames ** 2, axis=0)
        noise_energy = np.sum((reference_frames - estimate_frames) ** 2, axis=0)
        snr = librosa.power_to_db(signal_energy / noise_energy)
        source_snrs.append(snr)
    return np.stack(source_snrs)

def get_last_checkpoint(model_name, evaluation_dir):
    checkpoint_dirs = sorted(glob.glob(f'{evaluation_dir}/{model_name}-*'))
    last_checkpoint_dir = os.path.basename(checkpoint_dirs[-1])
    _model_name, checkpoint = last_checkpoint_dir.split('-', 1)
    return checkpoint

def get_model_params(model_name, models, nmf, dataset_dir):
    '''
    Returns:
        (full_model_name, model_name, checkpoint, extracted_sources, multi_source)
    '''
    if nmf:
        return model_name, model_name, None, all_source_names, False
    else:
        if '-' in model_name:
            model_name, checkpoint = model_name.split('-', 1)
        else:
            checkpoint = get_last_checkpoint(model_name, dataset_dir)
            full_model_name = f'{model_name}-{checkpoint}'
        
        model_params = models.loc[model_name]
        extracted_sources = [source_abbrev_to_source_name[abbrev] for abbrev in model_params.extracted_sources]
        return full_model_name, model_name, checkpoint, extracted_sources, model_params.multi_source

def get_estimates_template(nmf, multi_source, chorale):
    if nmf:
        return '%s.wav'
    else:
        if multi_source:
            return f'%s/chorale_{chorale}_mix.wav_source.wav'
        else:
            return f'chorale_{chorale}_mix.wav_%s.wav'

def evaluate_test_dataset(test_dataset, models_to_evaluate, nmf):
    models = read_models(os.environ['RESULTS_SHEET_ID'])
    parts_dir = f'{dataset_base}/{test_dataset}/audio_mono'
    evaluation_dir = 'nmf_evaluation' if nmf else 'test'
    dataset_dir = f'{evaluation_dir}/{test_dataset}'
    if nmf:
        dataset_sr = 22050
    else:
        dataset_sr = dataset_sample_rates[test_dataset]
    frame_size_seconds = 1
    frame_size_samples = frame_size_seconds * dataset_sr
    print(f'Evaluating on dataset {test_dataset}, evaluation frame size: {frame_size_samples} samples.')

    chorale_reference_sources = {}
    for chorale in test_chorales:
        chorale_reference_sources[chorale] = read_sources(f'{parts_dir}/chorale_{chorale}_%s.wav', all_source_names, dataset_sr)

    Metrics = NamedTuple('Metrics', [
        ('sdr', np.ndarray),
        ('isr', np.ndarray),
        ('sir', np.ndarray),
        ('sar', np.ndarray),
    ])
    for model in models_to_evaluate:
        full_model_name, model_name, checkpoint, extracted_sources, multi_source = get_model_params(model, models, nmf, dataset_dir)
        print(f'Model {model}:')
        model_test_dir = f'{dataset_dir}/{full_model_name}'
        chorale_metrics: Dict[str, Metrics] = {}
        print(f'Evaluating sources: {", ".join(extracted_sources)}')
        model_source_indices = [all_source_names.index(s) for s in extracted_sources]
        for chorale in test_chorales:
            print(f'\tChorale {chorale}')
            reference_sources = chorale_reference_sources[chorale]
            estimates_template = f'{model_test_dir}/{chorale}/{get_estimates_template(nmf, multi_source, chorale)}'
            # Initialize estimates to random because `museval.estimate` raises an exception
            # if any estimate is all-zeros. However, we do want to supply all _reference_
            # sources in order to correctly calculate SIR (interference from other sources),
            # and the shape of `estimated_sources` and `reference_sources` must be identical.
            estimated_sources = np.random.uniform(-1, 1, reference_sources.shape)
            estimated_sources[model_source_indices] = read_sources(estimates_template, extracted_sources, dataset_sr)
            # Return shape of each metric: (nsrc, nwin)
            sdr, isr, sir, sar = museval.evaluate(reference_sources, estimated_sources, padding=False, win=frame_size_samples, hop=frame_size_samples)
            chorale_metrics[chorale] = Metrics(sdr, isr, sir, sar)
        
        chorale_source_metrics_dfs = []
        for chorale, metrics in chorale_metrics.items():
            for source, source_index in zip(extracted_sources, model_source_indices):
                columns = {
                    'model': model_name,
                    'checkpoint': checkpoint,
                    'chorale': chorale,
                    'source': source
                }
                for metric, values in metrics._asdict().items():
                    columns[metric] = values[source_index]

                df = pandas.DataFrame(columns)
                df.insert(3, 'frame', df.index)
                chorale_source_metrics_dfs.append(df)

        model_metrics = pandas.concat(chorale_source_metrics_dfs, ignore_index=True)
        output_path = f'{model_test_dir}/evaluation.csv'
        model_metrics.to_csv(output_path)

def main():
    parser = argparse.ArgumentParser(description='Evaluate source estimations for chorale test dataset.')
    parser.add_argument('models', metavar='MODEL', nargs='+',
                        help='Model to evaluate. If model name is given without checkpoint, last model checkpoint is evaluated.')
    parser.add_argument('--dataset', help='Test dataset name', required=True)
    parser.add_argument('--nmf', help='Evaluate NMF results', action='store_true')
    args = parser.parse_args()
    evaluate_test_dataset(args.dataset, args.models, args.nmf)

if __name__ == '__main__':
    main()