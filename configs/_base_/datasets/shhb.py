dataset = dict(
    name='SHHB',
    data_root='./processed_data/SHHB',
    img_norm_cfg=dict(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
    scale=8,
    crop_size=400,
    batch_size=8,
)