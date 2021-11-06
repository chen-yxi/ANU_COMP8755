# ANU_COMP8755
Artefact for COMP8755

Run on Visual Studio Code

Packages:\
Python: 3.8.5  \ 
Numpy: 1.21.2   \
Pandas:  1.1.3  \
Pytorch: 1.7.1 + cuda 10.1 \
einops: 0.3.0    \
matplotlib: 3.3.2  

You can modify the hyperparameters in config.py\
Change source signal params['src'] and target signal params['tgt'] in config.py firstly\
Signal Data are saved in folder named data and signal for each participants in the subfolder pxx\
Trained Models are saved in folder named models\
train.py is used to preprocess data and train model\
infer.py is used to provide results of translation by trained model\
Optim.py is the optimizer file\
model.py contains the transformer structure
