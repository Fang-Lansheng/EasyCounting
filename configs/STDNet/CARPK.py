_base_ = [
    '../../_base_/models/CSRNet.py',
    '../../_base_/datasets/carpk.py',
    '../../_base_/schedules/schedule_500.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    name='STDNet',
    num_blocks=4,
    use_bn=False,
)

dataset = dict(
    name='CARPK',
    data_root='./processed_data/CARPK',
    batch_size=1,
    scale=5,           # aka. TIME_STEP (for this dataset)
)

optimizer = dict(
    type='Adam',
    lr=1e-4,
)

runner = dict(
    print_freq=40,
    max_epochs=250,
    num_workers=4,
)
