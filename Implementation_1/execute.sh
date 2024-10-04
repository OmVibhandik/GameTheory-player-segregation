set -e
echo "Downloading Dependencies"
pip install -r requirements.txt
echo "Starting the segregation script"
python main.py
echo "Segregation Finished"