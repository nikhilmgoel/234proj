import tensorflow as tf

class config():

    env_name="NavigateEnv-v0"

    record = True
    
    # model and training config
    num_batches         = 1000 # number of batches trained on 
    batch_size          = 20000 # number of steps used to compute each policy update
    max_ep_len          = 1000 # maximum episode length
    learning_rate       = 3e-2
    gamma               = 0.1 # the discount factor
    use_baseline        = True 
    normalize_advantage = True

    # parameters for the policy and baseline models
    n_layers            = 2 
    layer_size          = 32 
    activation          =tf.nn.relu 

    # output config
    output_path  = "results/" + env_name + "/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path 
    record_freq = 5
    summary_freq = 1


    # since we start new episodes for each batch
    assert max_ep_len <= batch_size
    if max_ep_len < 0: max_ep_len = batch_size