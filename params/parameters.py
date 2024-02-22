
#############################
# Training Parameters
#############################
num_of_epochs = 600
audio_sampling_rate = 16000

#############################
# Model Parameters
#############################
model_prefix = 'happy'
model_dir =f"/home/maxxies/Emodio/Models/{model_prefix}"


#############################
# Datsets Parameters
#############################
train_A_dir = '/home/maxxies/Emodio/Audios/neutral'
train_B_dir = f"/home/maxxies/Emodio/Audios/{model_prefix}"


#############################
# Training logs Parameters
#############################
train_logs__dir ='/home/maxxies/Emodio/Train_Logs'

#############################
# Output Parameters
#############################
norm_dir = '/home/maxxies/Emodio/Normalizations'
output_dir = '/home/maxxies/Emodio/output'
log_dir = '/home/maxxies/Emodio/Tensorboard_Logs'