_base_ = [
    '../_base_/models/PSNet.py',
    '../_base_/datasets/shha.py',
    '../_base_/schedules/schedule_500.py',
    '../_base_/default_runtime.py'
]

dataset = dict(
    data_root='./processed_data/SHHA',
    batch_size=8,
)

optimizer = dict(
    lr=1e-4,
)

runner = dict(
    print_freq=5,
    num_workers=4,
)
