
#############################
# Training Parameters
#############################
num_of_epochs = 600
audio_sampling_rate = 16000

#############################
# Model Parameters
#############################
# model_prefix = 'happy'
model_dir =f"/home/jupyter/Models/{model_prefix}"


#############################
# Datsets Parameters
#############################
train_A_dir = '/home/jupyter/Audios/neutral'
train_B_dir = f"/home/jupyter/Audios/{model_prefix}"


#############################
# Training logs Parameters
#############################
train_logs__dir ='/home/jupyter/Train_Logs'

#############################
# Output Parameters
#############################
norm_dir = '/home/jupyter/Normalizations'
output_dir = '/home/jupyter/output'
log_dir = '/home/jupyter/Tensorboard_Logs'