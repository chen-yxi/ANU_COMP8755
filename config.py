'''
Parameters of this
'''
params = dict()

# Dataset: [HR,EDA,BVP]
params['src'] = 'HR'
params['tgt'] = 'EDA'


# Parameters of Training
params['input_window']  = 100
params['output_window'] = 1
params['batch_size']    = 5

# Model Path
params['model_path'] = 'models/{0}2{1}-{2}windows.pth'\
    .format(params['src'],params['tgt'],params['input_window'])

# Parameters of Test
Freq_dict = {'HR':1,'EDA':4,'BVP':64}
params['Freq'] = Freq_dict[params['tgt']]
params['seq']  = 500
params['result_path'] = '{0}2{1}results{2}/'\
    .format(params['src'],params['tgt'],params['seq'])