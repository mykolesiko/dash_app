```
|   .gitignore    
|   app.py   
|   Dockerfile  
|   predict.py   
|   README.md   
|   requirements.txt   
|   requirements_local.txt   
|   temp.txt   
|      
+---data_test   
|       dogs.JPEG     
|       dogs.JPEG_mask   
|          
+---eda   
|       dogs.ipynb   
|       README.md   
|       
+---models   
|       best_resnet.pth   
|       dogs_best.pth   
|       dogs_segmented_best_881.pth   
|         
+---scripts   
|       get_data.sh   
|          
\---src  
    |   constants.py  
    |   __init__.py  
    |     
    +---datasets  
    |       datasets.py  
    |         
    +---model  
    |       constants.py  
    |       model.py  
    |       __init__.py  
    |         
    +---predict  
    |       main.py  
    |       predict.py  
    |       __init__.py  
    |       
    \---transforms  
            constants.py  
            transforms.py  
            __init__.py  
```

   
In order to Run the flask application with dogs
Assuming that current directory is root of git repository you should make next steps
1) build the docker image:
```

    docker build  -t dog_app .
```

2) run the application:
```
    docker run -p 8000:8000 dog_app
```
3) try it on http://localhost:8000

 In order to predict the brand of the dog on the image

1) Create and activate new environment:
```
    Installation (for Windows):
    python -m venv .venv
    .venv\Scripts\activate.bat
    pip install -r requirements_loc.txt

    Installation (for Linux):
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements_loc.txt
```
2) try
```
    python predict.py --data_dir .\data_test --model_dir .\models --image_name dogs.JPEG
```

Link to notebook
https://colab.research.google.com/drive/1mpQiKNHI-rPlS6iytC_t0h5W8K1v0DX5?usp=sharing#scrollTo=A-X1treQNqTB


Link to segmented and clipped data:
https://drive.google.com/drive/folders/1z1-wInSImW_UBQjQPRANjL-sfICnwDX4?usp=sharing

