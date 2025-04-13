import os
import argparse
from PIL import Image
from SODDCNet import *
from SODDCNetXL import *
from torchvision import transforms
import tqdm

MODEL_WIDTH, MODEL_HEIGHT = 384, 384

def load_model(model_size, cuda=None):
	print("Loading model...")
	
	if model_size == 'L':
		# Large Model
		model = SODDCNet(3, 1, use_contour = True, 
			deep_supervision = True, factorw = 8, factorwo = 4, img_res = MODEL_HEIGHT, dilation_rates = [[1, 1, 1, 1], [1, 1, 1]], 
			conv_sizes = [[9, 7, 5, 3], [9, 7, 5]], levels = [1, 1], conv_levels = [4, 3]
		)
		model.to(cuda)
		checkpoint = torch.load('checkpoints/SODDCNetL.pt', map_location=cuda)

	elif model_size == 'XL':
		## X-Large Model
		model = SODDCNetXL(3, 1, use_contour = True, 
			deep_supervision = True, factorw = 8, factorwo = 4, img_res = MODEL_HEIGHT, dilation_rates = [[1, 1, 1, 1], [1, 1, 1]], 
			conv_sizes = [[9, 7, 5, 3], [9, 7, 5]], levels = [1, 1], conv_levels = [4, 3]
		)
		model.to(cuda)
		checkpoint = torch.load('checkpoints/SODDCNetXL.pt', map_location=cuda)

	else:
		raise NotImplementedError

	
	model.load_state_dict(checkpoint['model_state_dict'], strict = True)
	model.eval()
	return model

def run_inference(model, image, cuda):
	original_width, original_height = image.size

	input_resized = transforms.Compose([transforms.Resize((MODEL_HEIGHT, MODEL_WIDTH)), transforms.ToTensor()])(image)

	contours, prediction = model(input_resized.unsqueeze(0).to(cuda))
	prediction = prediction[-1]
	pred = torch.sigmoid(prediction)
	pred = nn.Upsample(size=(original_height, original_width), mode='bilinear',align_corners=False)(pred)
	output_final = transforms.ToPILImage()(pred.squeeze(0))

	return output_final

def process_single_image(model, input_path, cuda, display=False):

	if not os.path.isfile(input_path):
		raise ValueError(f"Single image mode: '{input_path}' is not a valid file.")

	image = Image.open(input_path)
	prediction = run_inference(model, image, cuda)

	if display:
		prediction.show()
	else:
		# Save with the name of the image followed by '_prediction'
		base_name = os.path.splitext(os.path.basename(input_path))[0]
		save_name = f"{base_name}_prediction.png"
		prediction.save(save_name)
		print(f"Saved prediction to '{save_name}'")

def process_folder(model, input_path, output_dir, cuda):
	"""
	Process all images in a directory. Saves predictions in 'output_dir'.
	"""
	if not os.path.exists(input_path) or not os.path.isdir(input_path):
		raise ValueError(f"Folder mode: '{input_path}' is not a valid directory.")

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	for file_name in tqdm.tqdm(os.listdir(input_path)):
		file_path = os.path.join(input_path, file_name)
		
		if os.path.isfile(file_path) and file_name.lower().endswith((".jpg", ".jpeg", ".png")):
			image = Image.open(file_path)
			prediction = run_inference(model, image, cuda)

			base_name = os.path.splitext(file_name)[0]
			out_file = f"{base_name}.png"
			out_path = os.path.join(output_dir, out_file)

			prediction.save(out_path)

def main():
	parser = argparse.ArgumentParser(description="Run model inference on an image or a folder of images.")
	
	parser.add_argument(
		"--mode",
		type=str,
		choices=["single", "folder"],
		required=True,
		help="Mode of operation: 'single' for a single image, 'folder' for a directory of images."
	)
	parser.add_argument(
		"--input_path",
		type=str,
		required=True,
		help="Path to a single image (if mode=single) or to a folder (if mode=folder)."
	)
	parser.add_argument(
		"--display",
		action="store_true",
		help="(Only for single image mode) Display the prediction instead of saving it."
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default=None,
		help="(Only for folder mode) Directory to save the output predictions."
	)
	parser.add_argument(
		"--model_size",
		type=str,
		default='L',
		help="Either L or XL."
	)

	args = parser.parse_args()
	device = 0 ## cpu
	cuda = torch.device("cuda:" + str(device) if torch.cuda.is_available() else "cpu")

	print(f"Model Size  = {args.model_size}")
	print(f"Mode         = {args.mode}")
	print(f"Input Path = {args.input_path}")
	print(f"Output Directory   = {args.output_dir}")

	model = load_model(args.model_size, cuda)

	if args.mode == "single":
		process_single_image(model, args.input_path, cuda, display=args.display)
	elif args.mode == "folder":
		if not args.output_dir:
			raise ValueError("In folder mode, --output_dir is required.")
		process_folder(model, args.input_path, args.output_dir, cuda)

if __name__ == "__main__":
	main()
