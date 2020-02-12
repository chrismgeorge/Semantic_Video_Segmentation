scp -i exp_machine_2.pem ./Semantic_Video_Segmentation/setup.sh ubuntu@ec2-52-14-252-136.us-east-2.compute.amazonaws.com:~/
scp -i exp_machine_2.pem -r ./videos/ ubuntu@ec2-52-14-252-136.us-east-2.compute.amazonaws.com:~/
