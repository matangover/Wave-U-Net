import tensorflow as tf

import Utils

def independent_outputs(featuremap, source_names, num_channels, filter_width, padding, activation, scores):
    outputs = dict()
    for name in source_names:
        if scores:
            with tf.variable_scope('concat_score_' + name):
                score = scores[name]
                featuremap_for_source = tf.concat([featuremap, score], axis=2)
        else:
            featuremap_for_source = featuremap
    
        outputs[name] = tf.layers.conv1d(featuremap_for_source, num_channels, filter_width, activation=activation, padding=padding, name=name)
    return outputs

def difference_output(input_mix, featuremap, source_names, num_channels, filter_width, padding, activation, training):
    outputs = {}
    for name in source_names[:-1]:
        outputs[name] = tf.layers.conv1d(featuremap, num_channels, filter_width, activation=activation, padding=padding, name=name)

    # Compute last source based on the others
    with tf.variable_scope(source_names[-1]):
        sum_of_all_sources_but_last = tf.math.add_n(outputs.values())
        cropped_input = Utils.crop(input_mix, featuremap.get_shape().as_list(), match_feature_dim=False)
        last_source = Utils.crop(cropped_input, sum_of_all_sources_but_last.get_shape().as_list()) - sum_of_all_sources_but_last
        last_source = Utils.AudioClip(last_source, training)
    outputs[source_names[-1]] = last_source
    return outputs