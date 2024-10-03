
test_config = {
    "conf_threshold" : 0.41,
    "use_nms" : True,
    "as_lanes" : True,
    "nms_thres" : 50,
    "nms_topk" : 4,
    "ori_img_w" : 1640,
    "ori_img_h" : 590,
    "cut_height" : 270,
}

resnet18_neck = {
    "in_channels" : [64, 128, 256],
    "out_channels" : 64,
    "num_outs" : 3
}