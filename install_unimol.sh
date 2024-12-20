# Get python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11
sudo apt install python3-pip
sudo apt install python3.11-dev
sudo apt install python3.11-venv

# Create virtualenv
python3.11 -m venv venv
source venv/bin/activate
pip3 install -r unimol_requirements.txt

# Install unimol_tools
cd unimol_tools
pip install -e .
cd ..

# Install unicore
git clone https://github.com/dptech-corp/Uni-Core.git
cd Uni-Core
pip install -e .
cd ..

# Download checkpoints
wget https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_all_h_220816.pt
wget https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt