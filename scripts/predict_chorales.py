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
    run_id = None
    if run_id is None:
        raise ValueError('run_id required')

    model_dir = '%s/%s' % (checkpoint_base, run_id)
    checkpoint = get_last_checkpoint(model_dir)
    mixture_dataset = 'chorales_synth'

    output_path = 'evaluation/%s/%s-%s' % (mixture_dataset, run_id, checkpoint)
    
    model_path = '%s/%s-%s' % (model_dir, run_id, checkpoint)
    chorales_path = 'datasets/' + mixture_dataset

def get_last_checkpoint(model_dir):
    index_files = sorted(glob.glob('%s/*.index' % model_dir))
    last_checkpoint_filename = os.path.basename(index_files[-1])
    return os.path.splitext(last_checkpoint_filename)[0].split('-')[-1]

def predict_chorale(model_config, chorale, model_path, output_path):
    mix_path = chorale['mix']
    chorale_number = os.path.basename(mix_path).split('_')[1]
    output_path = os.path.join(output_path, chorale_number)

    if model_config['multiple_source_training']:
        for source in model_config['source_names']:
            print('\t' + source)
            source_output_path = os.path.join(output_path, source)
            score_filenames = {'source': chorale[source + '_score']}
            Evaluate.produce_source_estimates(model_config, model_path, mix_path, source_output_path, score_filenames)

    else:
        if model_config['score_informed']:
            score_filenames = {
                source_name: chorale[source_name + '_score']
                for source_name in model_config["source_names"]}
        else:
            score_filenames = {}
        Evaluate.produce_source_estimates(model_config, model_path, mix_path, output_path, score_filenames)
    
@ex.automain
def main(cfg, model_path, output_path, chorales_path):
    model_config = cfg["model_config"]
    print(f'Running prediction for model: {model_path}')
    print(f'Reading chorale mixtures from {chorales_path}')
    chorales = Datasets.getSynthesizedChorales(chorales_path, model_config["score_informed"])
    # Dataset.preprocess_dataset divides chorales into partitions, test partition starts after 320 chorales.
    test_chorales = chorales[320:]
    print('Saving outputs to: %s' % output_path)

    for i, chorale in enumerate(test_chorales):
        print(f'Predicting chorale {i+1} out of {len(test_chorales)}...')
        predict_chorale(model_config, chorale, model_path, output_path)
    