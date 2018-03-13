class config():
    # env config
    render_train     = False
    render_test      = True
    env_name         = "gym-navigate"
    overwrite_render = True
    record           = True
    high             = 255.

    # output config
    output_path  = "results/baseline_network/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path + "monitor/"

    # model and training config that need manual setting
    num_scenes         = 3
    video_duration     = 9031
    episode_duration   = 967             # how long we want each test/train episode to last
    num_episodes       = 28              # manually divide video_duration by episode_duration and multiply by num_videos
    num_episodes_train = 20
    num_episodes_test  = 8

    # other model and training config
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 250000
    log_freq          = 50
    eval_freq         = 250000
    record_freq       = 250000
    soft_epsilon      = 0.05

    # nature paper hyper params
    nsteps_train       = 5000000
    batch_size         = 32
    buffer_size        = 1000000
    target_update_freq = 10000
    gamma              = 0.99
    learning_freq      = 8
    state_history      = 8
    skip_frame         = 8
    lr_begin           = 0.00025
    lr_end             = 0.00005
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = 1000000
    learning_start     = 50000
