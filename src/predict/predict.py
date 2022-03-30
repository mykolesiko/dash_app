from src.model import ModelPredict
from src.transforms import val_transforms
import torch

def make_predict(data_dir: str, model_dir:str, model_name : str, image_name:str):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model_to_predict  =   ModelPredict(model_dir, data_dir, device, val_transforms)
    #image_file = glob.glob(os.path.join(data_dir, image_name))
    #image_name = np.random.choice(image_name)
    prediction = model_to_predict.predict(image_name, model_name)
    return prediction
