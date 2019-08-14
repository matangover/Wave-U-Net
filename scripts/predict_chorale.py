from sacred import Experiment
from Config import config_ingredient
import Evaluate
import os
import Datasets
from pathlib import Path
import glob

ex = Experiment('Waveunet Prediction', ingredients=[config_ingredient])

@ex.config
def cfg():
    checkpoint_base = 'checkpoints'
    dataset = 'chorales_synth'
    run_id = None
    if run_id is None:
        raise ValueError('run_id required')

    model_dir = '%s/%s' % (checkpoint_base, run_id)
    checkpoint = get_last_checkpoint(model_dir)
    input_base = f'datasets/{dataset}/mix'
    input_pattern = 'chorale_%s_mix.wav'
    chorale_number = 350
    midi_base = f'datasets/{dataset}/midi'

    source = None
    suffix = None

    output_path = 'outputs_test/%s-%s' % (run_id, checkpoint)
    
    if source is not None:
        output_path += '-%s' % source

    if suffix is not None:
        output_path += '-%s' % suffix

    model_path = '%s/%s-%s' % (model_dir, run_id, checkpoint)
    input_path = '%s/%s' % (input_base, input_pattern % chorale_number)

@ex.capture
def get_score_filename(source_name, chorale_number, midi_base):
    return str(Path(midi_base) / ('chorale_%03d_%s.mid' % (chorale_number, source_name)))

def get_last_checkpoint(model_dir):
    index_files = sorted(glob.glob('%s/*.index' % model_dir))
    last_checkpoint_filename = os.path.basename(index_files[-1])
    return os.path.splitext(last_checkpoint_filename)[0].split('-')[-1]

@ex.automain
def main(cfg, model_path, input_path, output_path, source):
    model_config = cfg["model_config"]
    if source is None:
        if model_config['multiple_source_training']:
            raise ValueError('Please specify which source to extract: python predict_chorale.py with source=<source>')
        score_filenames = {
            source_name: get_score_filename(source_name)
            for source_name in model_config["source_names"]}
    else:
        score_filenames = {'source': get_score_filename(source)}
    Evaluate.produce_source_estimates(model_config, model_path, input_path, output_path, score_filenames)
    print('Outputs saved to: %s' % output_path)