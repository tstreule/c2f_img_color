from src.utils.image import *
from src.utils.image import load_image
from src.agent import C2FImageGANAgent

def run_test_on_params(agent, image, min_ax_size, shrink_size, max_c2f_depth):
    agent.min_ax_size = min_ax_size
    agent.shrink_size = shrink_size
    agent.shrink_size = max_c2f_depth

    name = "min_ax_size" + str(min_ax_size) + "shrink_size" + str(shrink_size) + "_max_c2f_depth_" + str(max_c2f_depth) + ".jpg"

    agent.colorize_image(image).save(name)

checkpoint_path = r"checkpoints/gan_stage_1_epoch_10.pt"
picture_path = r"C:\Users\Dr. Paul von Immel\Desktop\test2.JPG"

agent = C2FImageGANAgent()
agent.load_model(checkpoint_path)
image = load_image(picture_path)

run_test_on_params(agent,image, 32,2.0,10)
run_test_on_params(agent,image, 32,1.5,10)
run_test_on_params(agent,image, 64,2.0,10)
run_test_on_params(agent,image, 64,1.5,10)
run_test_on_params(agent,image, 32,1.2,10)
