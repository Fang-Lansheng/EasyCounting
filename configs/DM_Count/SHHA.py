_base_ = [
    '../_base_/models/DM_Count.py',
    '../_base_/datasets/shha.py',
    '../_base_/schedules/schedule_500.py',
    '../_base_/default_runtime.py'
]

dataset = dict(
    data_root='./processed_data/SHHA',
    batch_size=16,
)

optimizer = dict(
    lr=1e-5,
)

runner = dict(
    print_freq=5,
)
