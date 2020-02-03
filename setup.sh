git clone --recursive https://github.com/NVIDIA/semantic-segmentation.git
git clone https://github.com/chrismgeorge/Semantic_Video_Segmentation
mv ./Semantic_Video_Segmentation/semantic-segmentation-helpers/video_2_jpg.py ./semantic-segmentation/
mv ./Semantic_Video_Segmentation/semantic-segmentation-helpers/demo_folder.py ./semantic-segmentation/
cd semantic-segmentation
pip install gdown
mkdir ./pretrained_model/
cd pretrained_model/
gdown https://drive.google.com/uc?id=1P4kPaMY-SmQ3yPJQTJ7xMGAB_Su-1zTl
cd ..

