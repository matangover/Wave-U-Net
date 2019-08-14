from pathlib import Path

checkpoint_dir = Path('checkpoints')

def clean_checkpoints() -> None:
    run_dirs = (d for d in checkpoint_dir.iterdir() if d.is_dir())
    for run_dir in run_dirs:
        print(f'Cleaning {run_dir}')
        clean_run_checkpoints(run_dir)

def clean_run_checkpoints(run_dir: Path) -> None:
    run_id = run_dir.name
    # Get sorted checkpoint numbers. Example checkpoint file name: 028-42000.index
    # where 028 is run id and 42000 is checkpoint number.
    checkpoints = sorted(int(f.stem.split('-')[1]) for f in run_dir.glob('*.index'))
    checkpoint_indices_to_keep = {0, len(checkpoints) // 3, len(checkpoints) * 2 // 3, len(checkpoints) - 1}
    print(f'\tKeeping {len(checkpoint_indices_to_keep)} checkpoints out of {len(checkpoints)}')
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        checkpoint_files = list(run_dir.glob(f'{run_id}-{checkpoint}.*'))
        if checkpoint_index not in checkpoint_indices_to_keep:
            target_dir = checkpoint_dir.parent / 'removed' / run_id
            for checkpoint_file in checkpoint_files:
                target_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_file.rename(target_dir / checkpoint_file.name)

if __name__ == '__main__':
    clean_checkpoints()