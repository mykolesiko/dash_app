In order to Run the flask application with dogs
Assuming that current directory is root of git repository you should make next steps
1) build the docker image:

docker build  -t dog_app .

2) run the application:

docker run -p 8000:8000 dog_app

3) try it on http://localhost:8000

In order to predict the brand of the dog on the image

1) Create and activate new environment:

Installation (for Windows):
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements_loc.txt

Installation (for Linux):
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_loc.txt

2) try
python predict.py --data_dir .\data_test --model_dir .\models --image_name dogs.JPEG


