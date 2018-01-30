import gym
import numpy as np
import tensorflow as tf
import dsr_pong.constants as const
import dsr_pong.common.dsr as dsr
from dsr_pong.common.atari_wrappers import wrap_deepmind
from dsr_pong.common.replay import ReplayBuffer
import collections
import time
from tqdm import tqdm
import argparse


class EnvTimer():
    def __init__(self):
        #Variables for timing
        self.last_env_timings = collections.deque(maxlen= 100)
        self.last_time_stamp = time.clock()

    def time_env_step(self):
        cur_time = time.clock()
        timing = cur_time - self.last_time_stamp
        self.last_env_timings.append(timing)
        self.last_time_stamp = cur_time
        return np.mean(self.last_env_timings)

def anneal_epsilon(t,T,eps_0,eps_T):
    if t > T:
        # return the minimum epsilon after T is reached
        return eps_T
    else:
        #compute the linear annealed epsilon
        ratio = float(t) / T
        distance = eps_T - eps_0
        return eps_0 + ratio * distance




def train_network():

    #only allocate as much memory as needed
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction= 0.25)

    # Learning alogrithm from the paper
    # ---------------------------------


    # Do gym initialization first
    env = gym.make(const.ENV_NAME)
    env = wrap_deepmind(env)
    env.reset()

    print("Action space: %s" % env.action_space)
    numActions = env.action_space.n

    # Initialize the replay memory D
    D = ReplayBuffer(const.REPLAY_BUFFER_SIZE,frame_history_len= 4)

    # Configure tensorboard
    summary_writer = tf.summary.FileWriter(const.LOG_DIR)

    # Initialize the step count
    step_count = 0
    episodeCount = 0
    # Initialize the network
    q_net = dsr.DoubleDSR(number_of_actions= numActions,feature_size= const.FEATURE_DIM)

    #choose the right optimizer
    if const.OPTIMIZER == const.Optimizers.RMS:
        #RMS PROP
        optimizer_model = tf.train.RMSPropOptimizer(learning_rate= const.LEARNING_RATE,momentum= 0.95,epsilon=0.01)
        optimizer_model = minimize_and_clip(optimizer_model,q_net.current_net.loss_a + q_net.current_net.loss_r,var_list= None)
        optimizer_sr = tf.train.RMSPropOptimizer(learning_rate= const.LEARNING_RATE, momentum= 0.95, epsilon= 0.01).minimize(q_net.current_net.loss_m)
    elif const.OPTIMIZER == const.Optimizers.MOMENTUM:
        #MOMENTUM
        optimizer_model = tf.train.MomentumOptimizer(learning_rate= const.LEARNING_RATE, momentum= 0.95)
        optimizer_model = minimize_and_clip(optimizer_model,q_net.current_net.loss_a + q_net.current_net.loss_r, var_list= None)
        optimizer_sr = tf.train.MomentumOptimizer(learning_rate= const.LEARNING_RATE, momentum= 0.95).minimize(q_net.current_net.loss_m)
    elif const.OPTIMIZER == const.Optimizers.ADAM:
        #ADAM
        optimizer_model = tf.train.AdamOptimizer(learning_rate= const.LEARNING_RATE)
        optimizer_model = minimize_and_clip(optimizer_model, q_net.current_net.loss_a + q_net.current_net.loss_r,
                                            var_list=None)
        optimizer_sr = tf.train.AdamOptimizer(learning_rate= const.LEARNING_RATE).minimize(q_net.current_net.loss_m)
    elif const.OPTIMIZER == const.Optimizers.ADAGRAD:
        #ADAGRAD
        optimizer_model = tf.train.AdagradOptimizer(learning_rate= const.LEARNING_RATE)
        optimizer_model = minimize_and_clip(optimizer_model, q_net.current_net.loss_a + q_net.current_net.loss_r,
                                            var_list=None)
        optimizer_sr = tf.train.AdagradOptimizer(learning_rate= const.LEARNING_RATE).minimize(q_net.current_net.loss_m)


    update_target_network = q_net.switch()
    init_network = tf.global_variables_initializer()

    current_reward_steps = const.REWARD_TRAINING_STEPS

    # Start the tensorflow session
    with tf.Session(config= tf.ConfigProto(gpu_options = gpu_options)) as sess:
        saver = tf.train.Saver()
        total_loss_summary = tf.summary.scalar("loss", q_net.current_net.total_loss)
        r_loss_summary = tf.summary.scalar("R_Loss_Summary", q_net.current_net.loss_r)
        a_loss_summary = tf.summary.scalar("A_Loss_Summary", q_net.current_net.loss_a)
        m_loss_summary = tf.summary.scalar("M_Loss_Summary", q_net.current_net.loss_m)

        # finalize the graph before training starts
        sess.graph.as_default()
        sess.graph.finalize()

        sess.run(init_network)
        saver.save(sess, const.LOG_DIR + '/model')

        # log the graph to the summary
        summary_writer.add_graph(sess.graph)

        env_timer = EnvTimer()

        total_rewards = []
        training_terminated = False

        while not training_terminated:

            # Initialize the environment
            s , done = env.reset() , False
            episode_reward = 0

            print("New Episode started.")

            while not done and not training_terminated:

                #add the current state to the replay buffer
                s_index = D.store_frame(s)

                #compute the current epsilon
                eps_t = max([step_count - const.TRAINING_START, 0])
                eps = anneal_epsilon(t = eps_t, T = const.EPSILON_END_T,eps_0= const.EPSILON_START, eps_T= const.EPSILON_END)

                #write the epsilon to the summary
                summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag = "epsilon", simple_value= eps)]),step_count)

                # Do epsilon greedy exploration
                if np.random.rand() < eps:
                    # choose an action randomly
                    a = env.action_space.sample()
                else:
                    stacked_states = D.encode_recent_observation()

                    # query network for the best (known) action
                    fd = {q_net.current_net.X: [stacked_states]}
                    [a] = sess.run(fetches= q_net.current_net.action ,feed_dict= fd)

                # Do a step in the real environment
                s_, r, done, _ = env.step(a)

                #Update stats
                episode_reward += r

                # Store the new experience to the replay memory
                D.store_effect(s_index,a,r,done)

                # increase the step count
                step_count += 1

                #check if it is a reward training step
                if step_count >= const.REWARD_TRAINING_START and step_count % const.REWARD_TRAINING_FREQ == 0:
                    tqdm.write("Train the reward branch for %d steps:" % current_reward_steps)
                    for _ in tqdm(range(current_reward_steps)):
                        #sample a reward batch
                        batch_s, batch_a, batch_r, batch_s_, batch_done = D.sample_priorized(
                            batch_size=const.BATCH_SIZE, prio_size=const.PRIO_BATCH_SIZE)

                        feed_dict = {q_net.current_net.X : batch_s,
                                     q_net.current_net.observed_reward : batch_r}

                        #train the reward branch
                        sess.run(fetches= [optimizer_model],feed_dict= feed_dict)

                    #anneal the reward training steps
                    current_reward_steps = current_reward_steps // 2


                if step_count >= const.TRAINING_START and step_count % const.PLAY_STEPS == 0:
                    #sample a new mini batch from the replay buffers
                    batch_s, batch_a, batch_r, batch_s_, batch_done = D.sample_priorized(batch_size= const.BATCH_SIZE,prio_size= const.PRIO_BATCH_SIZE)

                    #Query network for consecutive action a_
                    [cons_actions] = sess.run([q_net.current_net.action],feed_dict= {q_net.current_net.X : batch_s_})

                    #Check if all consecutive actions are the same
                    if np.all(cons_actions == cons_actions[0]):
                        print("All actions are the same.")
                    else:
                        print("Different actions.")

                    #Query the SF targets from the previous network
                    pseudo_targets = sess.run(q_net.prev_net.succ_feature_representations, feed_dict= {q_net.prev_net.X : batch_s_})

                    #take the consecutive action and if it is a terminal state into account
                    targets = [pseudo_targets[i,cons_actions[i],:] if not batch_done[i] else np.zeros(const.FEATURE_DIM) for i in range(const.BATCH_SIZE)]

                    #Do a training step
                    fd = {q_net.current_net.X : batch_s,
                          q_net.current_net.observed_reward : batch_r,
                          q_net.current_net.observed_action : batch_a,
                          q_net.current_net.succ_feature_target : targets}

                    #optimize the model first
                    _ = sess.run(fetches= optimizer_model, feed_dict= fd)

                    fetches = [optimizer_sr, q_net.current_net.total_loss,
                               total_loss_summary,
                               r_loss_summary,
                               a_loss_summary,
                               m_loss_summary]

                    _, loss, t_sum_v, r_sum_v, a_sum_v , m_sum_v = sess.run(fetches= fetches, feed_dict=fd)
                    print("Loss of %f for training batch. Action %d choosen." % (loss, a))

                    #write the summary
                    summary_writer.add_summary(t_sum_v,step_count)
                    summary_writer.add_summary(r_sum_v,step_count)
                    summary_writer.add_summary(a_sum_v,step_count)
                    summary_writer.add_summary(m_sum_v,step_count)
                    summary_writer.flush()

                # Check if the networks need to be switched
                if step_count % const.NETWORK_SYNC_STEPS == 0:
                    print("Sync.")
                    sess.run(update_target_network)

                #Check if the network needs to be saved
                if step_count % const.SAVE_INTERVAL == 0:
                    saver.save(sess,const.LOG_DIR + "/model",step_count)

                #Check if it is time for evaluation
                if step_count >= const.TRAINING_START and step_count % const.EVALUATION_INTERVAL == 0:
                    stats = evaluate(q_net,sess)
                    eval_mean = np.mean(stats)
                    summary_writer.add_summary(
                        tf.Summary(value=[tf.Summary.Value(tag="evaluation_score", simple_value= eval_mean)]), step_count)


                #Meassure timing of the step
                step_time = env_timer.time_env_step()
                summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="speed", simple_value= 1 / step_time)]), step_count)

                s = s_

            print("Episode terminated. Achieved reward: %d" % episode_reward)
            episodeCount += 1
            #update stats
            total_rewards.append(episode_reward)
            mean = np.mean(total_rewards[-100:])
            print("Mean of the last 100 episodes: %f" % mean)
            summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="episode_rewards", simple_value=episode_reward)]), episodeCount)
            summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag="mean_reward_100", simple_value=mean)]), episodeCount)

            #Check if pong is solved and terminate training
            if mean > 18:
                training_terminated = True

def minimize_and_clip(optimizer, objective, var_list, clip_val=1):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    with tf.name_scope("clip_gradients"):
        gradients = optimizer.compute_gradients(objective, var_list=var_list)
        for i, (grad, var) in enumerate(gradients):
            if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
        return optimizer.apply_gradients(gradients)

def sample_mini_batch(buffer, prio_buffer):
    # Randomly sample a mini_batch from the replay memory
    batch_s, batch_a, batch_r, batch_s_, batch_done = buffer.sample(const.BATCH_SIZE - const.PRIO_BATCH_SIZE)

    # Randomly sample the prio batch
    prio_batch_s, prio_batch_a, prio_batch_r, prio_batch_s_, prio_batch_done = prio_buffer.sample(const.PRIO_BATCH_SIZE)

    # join the batches
    batch_s = np.concatenate([batch_s, prio_batch_s])
    batch_a = np.concatenate([batch_a, prio_batch_a])
    batch_r = np.concatenate([batch_r, prio_batch_r])
    batch_s_ = np.concatenate([batch_s_, prio_batch_s_])
    batch_done = np.concatenate([batch_done, prio_batch_done])

    # premutate the new batch
    perm = np.random.permutation(range(const.BATCH_SIZE))
    batch_s = batch_s[perm]
    batch_a = batch_a[perm]
    batch_r = batch_r[perm]
    batch_s_ = batch_s_[perm]
    batch_done = batch_done[perm]

    return batch_s, batch_a, batch_r, batch_s_, batch_done

def evaluate(q_net,sess):
    tqdm.write("Evaluation started.")

    # Do gym initialization first
    env = gym.make(const.ENV_NAME)
    env = wrap_deepmind(env)

    #use a replay buffer for state encoding (stacked frames)
    D = ReplayBuffer(size = 8,frame_history_len= 4)

    eval_stats = []

    for _ in tqdm(range(10)):
        s, done = env.reset(), False
        episode_reward = 0

        while not done:
            #store current state for encoding
            D.store_frame(s)

            #select an action greedy
            stacked_states = D.encode_recent_observation()
            fd = {q_net.current_net.X: [stacked_states]}
            [a] = sess.run(fetches=q_net.current_net.action, feed_dict=fd)

            #take that action
            s_ , r , done, _ = env.step(a)

            #track reward
            episode_reward += r

        #episode done
        eval_stats.append(episode_reward)

    tqdm.write("Evaluation done.")

    return eval_stats


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--logdir",default= const.LOG_DIR)
    parser.add_argument("--opt", default="adam")

    args = parser.parse_args()
    const.LOG_DIR = args.logdir

    if args.opt == "rms":
        const.OPTIMIZER = const.Optimizers.RMS
    elif args.opt == "adam":
        const.OPTIMIZER = const.Optimizers.ADAM
    elif args.opt == "adagrad":
        const.OPTIMIZER = const.Optimizers.ADAGRAD
    elif args.opt == "momentum":
        const.OPTIMIZER = const.Optimizers.MOMENTUM

    train_network()