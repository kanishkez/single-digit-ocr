# single-digit-ocr
just a project to get into computer vision 
no multi digit because no gpu :( 



How to Use / Test the Program

1. Prepare a test image
	•	The project is designed for single handwritten digits.
	•	Create or obtain a grayscale or RGB image containing one digit (0-9).
	•	If needed, resize or pad so the digit is centered and prominent.

2. Run inference

python digitinference.py --image path/to/your_digit_image.png

	•	This script will load the pretrained model (digit.pth) and output the predicted digit.
	•	You can also modify the script to pass alternative image paths or batch inputs.

3. Interpret results
	•	The script will print something like:

Predicted digit: 7

Note: I have added trained weights in this repo which are used in the inference script
