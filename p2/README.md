# Semantic_Video_Segmentation
Prototype 2 for Semantic Video Segmentation.

## semantic-segmentation
* Create an AWS EC2 instance using the ubuntu Deep Learning AMI
* SSH into the instance:
    * ssh -i exp_machine.pem -L 8000:localhost:8888 ubuntu@instance
* Update the instance name within the local_setup.sh file, then run:
    * sh local.sh
* Within the EC2 instance:
    * source activate pytorch_p36
    * sh setup.sh
* Run the inference command for each video.
    * cd semantic-segmentation
    * CUDA_VISIBLE_DEVICES=0 python demo_folder.py --demo-folder YOUR_FOLDER --snapshot ./pretrained_models/cityscapes_best.pth --save-dir YOUR_FOLDER
* Download the video!
    * scp -i exp_machine.pem ubuntu@instance:~/filePathRemote ./
