_base_ = [
    '../../_base_/models/CSRNet.py',
    '../../_base_/datasets/venice.py',
    '../../_base_/schedules/schedule_500.py',
    '../../_base_/default_runtime.py'
]

model = dict(
    name='STDNet',
    num_blocks=4,
    use_bn=False,
)

dataset = dict(
    name='Venice',
    data_root='./processed_data/VENICE',
    batch_size=1,
    scale=5,           # aka. TIME_STEP (for this dataset)
)

optimizer = dict(
    type='Adam',
    lr=1e-4,
)

runner = dict(
    print_freq=4,
    max_epochs=100,
    num_workers=4,
)
