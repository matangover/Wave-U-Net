import wave
from pathlib import Path
import time

def format_duration(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

dataset_base = 'data'
column = '%12s%25s%25s%25s'
print(column % ('', 'no. of chorales', 'total duration', 'avg. chorale duration'))
for dataset in ['chorales_synth']:
    mixes = sorted(Path(f'{dataset_base}/{dataset}/mix').iterdir())
    partitions = {
        'training': mixes[:270],
        'validation': mixes[270:320],
        'test': mixes[320:]
    }

    print('\n' + dataset + ':\n')
    dataset_total = 0
    for partition, partition_mixes in partitions.items():
        partition_total = 0
        for mix in partition_mixes:
            with wave.open(str(mix)) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / rate
                partition_total += duration

        print(column % (
            partition,
            len(partition_mixes),
            format_duration(partition_total),
            format_duration(partition_total / len(partition_mixes))
        ))
        dataset_total += partition_total
    
    print(column % (
        'total',
        len(mixes),
        format_duration(dataset_total),
        format_duration(dataset_total / len(mixes))
    ))

