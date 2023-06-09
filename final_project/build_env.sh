read -p "Please input the name of environement: " env_name
conda create --name $env_name python=3.8 -y
conda activate $env_name
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -r requirements.txt