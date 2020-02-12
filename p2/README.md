# Semantic_Video_Segmentation
A series of artistic uses of semantic video segmentation.

## semantic-segmentation
* Scripting
    * Create an AWS EC2 instance using the ubuntu Deep Learning AMI
    * SSH into the instance:
        * ssh -i exp_machine.pem -L 8000:localhost:8888 ubuntu@instance
    * Update the instance name within the local_setup.sh file, then run:
        * sh local.sh
    * Within the EC2 instance:
        * sh setup.sh
            * This will automatially run the video_2_jpg file and turn all of the videos into jpgs existing within the videos/video_name/ folder.
            * Also automatically fixes the apex issue.
    * Run the inference command for each video.
        * CUDA_VISIBLE_DEVICES=0 python demo_folder.py --demo-folder YOUR_FOLDER --snapshot ./pretrained_models/cityscapes_best.pth --save-dir YOUR_SAVE_DIR
    * Run the jpg_2_video to turn the color_mask stuff into video form.
        * python jpg_2_video.py VIDEO_NAME
    * Download the video!
        * scp -i exp_machine.pem ubuntu@instance:~/filePathRemote ./
