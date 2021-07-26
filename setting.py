#setting for training
reading_range = 60 #corresponding to the hyperparameter M in the paper
max_dist = 30 #corresponding the hyperparameter N in the paper
texture = 'polyphony' #'polyphony' or 'homophony'

epoch = 50
batch_size = 2048
pile = 100

train_folder = 'train/'
val_folder = 'val/'
model_file = 'my_model.h5'
log_file = 'training_log.txt'

#setting for evaluation 
window = 40 #corresponding to the hyperparameter S in the paper
overlap_min = 30 #corresponding to the hyperparameter O in the paper
overlap_pitch = 0

output_midi = True
method = 'pitch' #'pitch'(pitch proximity) or 'min'(minimal overlapping)
input_folder = 'val/'
output_folder = 'pred/'
metrics_file = 'result.csv'