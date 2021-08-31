python -m torch.distributed.launch --nproc_per_node=1 test_panoptic.py --meta metadata.bin --log_dir dummy config/SemanticKITTI.ini model/model_best.pth.tar /home/valada/mohan/kitti_lidar_projection/img/ val/
#--resume b5_baseline_8l2_2nditer/model_best.pth.tar --eval
