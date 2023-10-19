# python video_test.py -m './checkpoints/resnet101v2_custom_BITS_ep30_emore.h5' -s './datasets/test.mp4'
# python video_test.py -m './checkpoints/resnet18_custom_BITS_ep50_bs16_emore.h5' -s './datasets/me.mp4' -k './datasets/custom_BITS_aligned_112_112'
python video_test.py -m './checkpoints/adaface_ir101_webface12m_rgb_customdata.h5' -k './datasets/custom_BITS_aligned_112_112' -p 100
# python video_test.py -m './checkpoints/resnet18_custom_BITS_ep50_bs16_emore.h5'
# python video_test.py -m './checkpoints/resnet101v2_custom_emore.h5' -s './datasets/videoplayback.mp4'
# python face_detector.py -D './datasets/custom_BITS'