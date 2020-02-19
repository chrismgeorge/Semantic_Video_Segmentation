# Semantic_Video_Segmentation
Prototype 3 for Semantic Video Segmentation.

## Instructions
1. Create an AWS EC2 instance using the ubuntu Deep Learning AMI
2. SSH into the instance:
```
ssh -i exp_machine.pem -L 8000:localhost:8888 ubuntu@instance
```
3. Update the instance name within the local_setup.sh file, then run:
```
sh local.sh
```
4. Within the EC2 instance:
```
source activate pytorch_p36
```
```
sh setup.sh
```
5. Run the inference command for each video.
```
cd semantic-segmentation
```
```
CUDA_VISIBLE_DEVICES=0 python demo_folder.py --demo-folder YOUR_FOLDER --snapshot ./pretrained_models/cityscapes_best.pth --save-dir YOUR_FOLDER
```
6. Download the video or image sequence!
```
scp -i exp_machine.pem ubuntu@instance:~/filePathRemote ./
```
```
scp -i exp_machine.pem -r ubuntu@instance:~/folderPathRemote/ ./
```
