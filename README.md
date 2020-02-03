# Semantic_Video_Segmentation
A series of artistic uses of semantic video segmentation.

## semantic-segmentation
* Step-by-Step
    * Create an AWS EC2 instance using the ubuntu Deep Learning AMI
    * SSH into the system
    * git clone --recursive https://github.com/NVIDIA/semantic-segmentation.git
    * cd semantic-segmentation
    * Create a new folder for the checkpoint
        * mkdir ./pretrained_model/
    * Use gdown to pull the data from google drive
        * pip install gdown
        * gdown https://drive.google.com/uc?id=1P4kPaMY-SmQ3yPJQTJ7xMGAB_Su-1zTl
    * Move the content of the semantic-segmentation-helpers into the top-level directory
    * Create a videos folder, and move all of the videos you to use into said folder.
    * Run video_2_jpg.py to turn all of the videos into a series of jpgs.
        * The images will be stored in a newly created folder called ./videos/video_name/0000X.jpg
        * It will also automatically create an output folder for all of the inference time outputs of the model ./videos/video_name_segmented/
    * In the demo_folder.py file:
        * on line 71 add colorized.convert('RGB')...
    * Apex install fix
        * https://github.com/NVIDIA/apex/issues/116
    * Run the inference command for each folder of images:
        * CUDA_VISIBLE_DEVICES=0 python demo_folder.py --demo-folder YOUR_FOLDER --snapshot ./pretrained_models/cityscapes_best.pth --save-dir YOUR_SAVE_DIR
    * Each video will now have a folder with three different outputs per image.
        * Depending on our use case, we can convert the images into video now, or at a later time.
        * If they are being converted at a later time, we will still need the original image for it's FPS.
        * If we would just like to create the videos within the instance, we can run the jpg_2_video.py
* Scripting
    * In an effort to efficiently perform this task, there are a few scripts that I wrote that can increase the speed at which things are done.
    * Steps:
        * Create an AWS EC2 instance using the ubuntu Deep Learning AMI
        * SSH into the instance:
            * ssh -i exp_machine.pem -L 8000:localhost:8888 ubuntu@instance
        * Update the instance name within the local_setup.sh file, then run:
            * sh local_setup.sh
                * scp -i exp_machine.pem ./Semantic_Video_Segmentation/setup.sh ubuntu@instance:~/
        * Witin the EC2 instance:
            * sh setup.sh
                * This will automatially run the video_2_jpg file and turn all of the videos into jpgs existing within the videos/video_name/ folder.
