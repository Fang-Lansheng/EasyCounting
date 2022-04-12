_base_ = [
    '../_base_/models/DM_Count.py',
    '../_base_/datasets/qnrf.py',
    '../_base_/schedules/schedule_500.py',
    '../_base_/default_runtime.py'
]

dataset = dict(
    data_root='./processed_data/QNRF',
)

optimizer = dict(
    lr=1e-5,
)
