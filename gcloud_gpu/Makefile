download.sh:
	echo "====================== Downloading CUDA 8.0 ========================"
	echo "... ... ..."
	curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
	echo "================================================="
	echo "               Downloading done!                 "
	echo "================================================="

install.sh: download.sh
	sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
	sudo apt-get update
	rm cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
	sudo apt-get install cuda -y
	echo "================  Install Done! ================="
	echo "================================================="
	echo "            Checking for NVIDIA Card             "
	echo "================================================="
	nvidia-smi

SHELL := /bin/bash

set_env_variables: install.sh
	echo "================================================="
	echo "======= Setting up environment variables ========"
	echo "================================================="
	source ./env.sh
	source ~/.bashrc

cudnn_install: set_env_variables
	sudo apt-get install p7zip-full
	echo "=============  Downloading cuDNN for CUDA8.0 ============"
	curl -O https://storage.googleapis.com/cudnn51/cudnn-8.0-linux-x64-v5.1
	echo "===================  Download done ======================"
	7za x cudnn-8.0-linux-x64-v5.1
	sudo cp -r cuda/lib64/* /usr/local/cuda/lib64/
	sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
	rm -rf ~/cuda
	rm cudnn-8.0-linux-x64-v5.1

SHELL := /bin/bash

miniconda_install: cudnn_install
	echo "============ Installing Python =================="
	curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh
	echo " ============ Installation done! ================"
	source ~/.bashrc

SHELL := /bin/bash

dl_lib_install: 
	conda create --name dlgpu python=3.5
	source activate dlgpu
	echo " ========= Installing TensorFlow ================"
	export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp35-cp35m-linux_x86_64.whl
	conda install setuptools
	source ./pip.sh
	sudo ldconfig /usr/local/cuda/lib64

SHELL := /bin/bash

jupyter.sh:
	cd
	jupyter notebook --generate-config
	echo "c = get_config()" >> ~/.jupyter/jupyter_notebook_config.py
	echo "c.IPKernelApp.pylab = 'inline'" >> ~/.jupyter/jupyter_notebook_config.py
	echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py
	echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py


