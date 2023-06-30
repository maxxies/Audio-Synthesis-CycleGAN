#############################
# Training Parameters
#############################
num_of_epochs = 100
audio_sampling_rate = 16000

#############################
# Datsets Parameters
#############################
train_A_dir = 'gs://audio_dataset_bucket/Audios/neutral'
train_B_dir = 'gs://audio_dataset_bucket/Audios/happy'

#############################
# Model Parameters
#############################
model_prefix = 'happy'
model_dir =f"/home/jupyter/model/{model_prefix}"

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