import torch
from torchvision import transforms
from PIL import Image, ImageOps
from digit import OCRModel  

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Transform to match training preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Mapping indices to characters for EMNIST 'balanced' split
idx_to_char = {
    0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
    10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',
    21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',31:'V',
    32:'W',33:'X',34:'Y',35:'Z',36:'a',37:'b',38:'d',39:'e',40:'f',41:'g',42:'h',
    43:'n',44:'q',45:'r',46:'t'
}

def preprocess_image(img_path):
    image = Image.open(img_path).convert("L")
    # Invert colors if needed (black background, white text)
    image = ImageOps.invert(image)
    # Rotate and flip to match EMNIST orientation
    image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    image = transform(image).unsqueeze(0).to(device)
    return image

def predict_character(img_path, model):
    image = preprocess_image(img_path)
    model.eval()
    with torch.inference_mode():
        pred = model(image)
    pred_class_idx = pred.argmax(dim=1).item()
    return idx_to_char[pred_class_idx]

if __name__ == "__main__":
    model = OCRModel().to(device)
    model.load_state_dict(torch.load('ocr_emnist_balanced.pth', map_location=device))
    
    img_path ='/Users/kanishkk/Downloads/WhatsApp Image 2025-09-19 at 09.35.30.jpeg' 
    char = predict_character(img_path, model)
    print(f"The character is: {char}")
