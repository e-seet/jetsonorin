sudo docker run -d --runtime nvidia --network host --shm-size=8g -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/l4t-ml:r36.2.0-py3 bash -c "
pip3 uninstall -y pyparsing numpy && \
pip3 install 'numpy<2' pyparsing==3.1.2 ultralytics && \
bash start_training.sh > /workspace/docker_training.log 2>&1

tail -f docker_training.log