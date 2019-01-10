CUDA_VISIBLE_DEVICES="0" python ./fast-neural-style/stylize_frames.py -style_name 'crayon' -video_path '/home/vincentwu-cmlab/Downloads/DIP_final/Sky/content/sky1/Video1_HR.avi' -output_name 'Video1_HR'
CUDA_VISIBLE_DEVICES="0" python ./fast-neural-style/stylize_frames.py -style_name 'crayon' -video_path '/home/vincentwu-cmlab/Downloads/DIP_final/Sky/content/sky2/Video2_HR.avi' -output_name 'Video2_HR'
CUDA_VISIBLE_DEVICES="0" python ./fast-neural-style/stylize_frames.py -style_name 'fountainpen' -video_path '/home/vincentwu-cmlab/Downloads/DIP_final/Sky/content/sky1/Video1_HR.avi' -output_name 'Video1_HR'
CUDA_VISIBLE_DEVICES="0" python ./fast-neural-style/stylize_frames.py -style_name 'fountainpen' -video_path '/home/vincentwu-cmlab/Downloads/DIP_final/Sky/content/sky2/Video2_HR.avi' -output_name 'Video2_HR'
CUDA_VISIBLE_DEVICES="0" python ./fast-neural-style/stylize_frames.py -style_name 'ZaoWouKi' -video_path '/home/vincentwu-cmlab/Downloads/DIP_final/Sky/content/sky1/Video1_HR.avi' -output_name 'Video1_HR'
CUDA_VISIBLE_DEVICES="0" python ./fast-neural-style/stylize_frames.py -style_name 'ZaoWouKi' -video_path '/home/vincentwu-cmlab/Downloads/DIP_final/Sky/content/sky2/Video2_HR.avi' -output_name 'Video2_HR'

CUDA_VISIBLE_DEVICES="0" python test_pretrained.py -dataset Sky -task fast-neural-style/crayon -redo
CUDA_VISIBLE_DEVICES="0" python test_pretrained.py -dataset Sky -task fast-neural-style/fountainpen -redo
CUDA_VISIBLE_DEVICES="0" python test_pretrained.py -dataset Sky -task fast-neural-style/ZaoWouKi -redo