from src.distillation import *
import os

if __name__ == '__main__':
    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./imgs"):
        os.makedirs("./imgs")

    print(args)

    distilling()

    # load_sg_model("./models/Distilled_G_Epoch_10.pth")
    rand_faces()