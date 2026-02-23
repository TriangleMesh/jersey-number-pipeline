# run this scripty by running "source SetupEnv.sh"

conda init
echo y | conda create --name SoccerNet python=3.9
conda activate SoccerNet
pip install gdown
pip install tqdm
pip install SoccerNet
pip install pandas
pip install opencv-python

if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "use CPU version torch on MacOS"
    pip install torch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0
else
    echo "use GPU version torch"
    pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
fi

pip install numpy==1.26.4
if [[ "$(uname -s)" == "Linux" ]]; then
  echo "Current platform is Linux"
  echo y | sudo apt-get install libgl1
fi
python SetupSoccerNetDataset.py
python setup.py SoccerNet
