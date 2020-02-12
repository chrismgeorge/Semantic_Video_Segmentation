git clone --recursive https://github.com/NVIDIA/semantic-segmentation.git
git clone https://github.com/chrismgeorge/Semantic_Video_Segmentation
mv ./Semantic_Video_Segmentation/semantic-segmentation-helpers/p2/video_2_jpg.py ./semantic-segmentation/
mv ./Semantic_Video_Segmentation/semantic-segmentation-helpers/p2/jpg_2_video.py ./semantic-segmentation/
rm ./semantic-segmentation/demo_folder.py
mv ./Semantic_Video_Segmentation/semantic-segmentation-helpers/p2/demo_folder.py ./semantic-segmentation/
mv ./videos/ ./semantic-segmentation/
cd semantic-segmentation
pip install gdown
mkdir ./pretrained_models/
cd pretrained_models/
gdown https://drive.google.com/uc?id=1P4kPaMY-SmQ3yPJQTJ7xMGAB_Su-1zTl
cd ..
pip install opencv-python
python3 video_2_jpg.py
pip install opencv-python
pip uninstall apex
cd ..
git clone https://www.github.com/nvidia/apex
cd apex
python setup.py install
cd ../semantic-segmentation/
