from model.model import captcha_model, model_conv, model_resnet
from utils.arg_parsers import predict_arg_parser
from data.dataset import str_to_vec, lst_to_str
import torch
import torchvision.transforms as transforms
from PIL import Image
from utils.config_util import configGetter
import onnx

cfg = configGetter('DATASET')

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict(args):
    model = captcha_model.load_from_checkpoint(args.ckpt, model=model_resnet())
    model.eval()
    
    img = transform(Image.open(args.input))
    img = img.unsqueeze(0)
    y = model(img)
    y = y.permute(1, 0, 2)
    pred = y.argmax(dim=2)

    ans = lst_to_str(pred)
    print(ans)
    
    # Save the model's state dictionary
    # torch.save(model.state_dict(), "./captcha_model.pt")
    # It's optional to label the input and output layers
    # input_names = [ "actual_input" ] + [ "learned_%d" % i for i in range(16) ]
    # output_names = [ "output" ]
    # # Define a dummy input tensor with matching shape to the model's input shape
    # dummy_input = torch.randn(1, 3, cfg['CAPTCHA']['IMG_HEIGHT'], cfg['CAPTCHA']['IMG_WIDTH'])
    # torch.onnx.export(model,
    #                   dummy_input,
    #                   "captcha_model.onnx",
    #                   verbose=True,
    #                   input_names=input_names,
    #                   output_names=output_names)
    # print("Model saved successfully.")

    # # Load the ONNX model
    # onnx_model = onnx.load("captcha_model.onnx")
    # # Check that the model is well-formed
    # try:
    #     onnx.checker.check_model(onnx_model)
    # except onnx.checker.ValidationError as e:
    #     print("The model is invalid: %s" % e)
    # else:
    #     print("The model is valid!")

    return ans


if __name__ == "__main__":
    args = predict_arg_parser()
    predict(args)
    

