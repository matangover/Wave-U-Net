from sacred import Experiment
from Config import config_ingredient
import Utils
import Datasets

chorales_to_check = [6, 15, 161, 209, 271]

ex = Experiment('Wave-U-Net', ingredients=[config_ingredient])

def get_chorale(chorales, chorale_number):
    for chorale in chorales:
        if chorale['mix'].endswith('chorale_%03d_mix.wav' % chorale_number):
            return chorale
    
    raise ValueError('Chorale not found: %s' % chorale_number)

@ex.automain
def run(cfg):
    model_config = cfg['model_config']
    chorales = Datasets.getSynthesizedChorales(model_config["chorales_path"], model_config["score_informed"])
    for chorale_number in chorales_to_check:
        chorale = get_chorale(chorales, chorale_number)
        input_shape, output_shape = Utils.get_separator_shapes(model_config)
        pad_frames = (input_shape[1] - output_shape[1]) // 2
        Datasets.process_sample(chorale, pad_frames, cfg['model_config'])