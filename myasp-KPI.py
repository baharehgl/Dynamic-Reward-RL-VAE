def train_with_validation(env_full, env_full_test, qlearn_constructor, target_constructor,
                          vae_model,
                          num_episodes=300,
                          num_epoches=10,
                          validate_every=10,
                          patience=3,
                          discount_factor=0.96,
                          num_LP=200,
                          num_AL=1000):
    """
    env_full: EnvKPI over ALL KPIs
    env_full_test: identical EnvKPI (for final test)
    qlearn_constructor(scope): returns a fresh Q_Estimator_Nonlinear
    target_constructor(scope): ditto
    vae_model: pretrained VAE
    """
    # 1) Split indices
    total = env_full.datasetsize
    train_size = int(total * validation_separate_ratio)
    train_indices = list(range(train_size))
    val_indices   = list(range(train_size, total))

    # 2) Build train & val envs
    env_train = env_full
    env_train.datasetsize = train_size
    def train_reset(idx=None):
        # wrap reset to only see train KPIs
        if idx is None:
            idx = random.choice(train_indices)
        return env_train.reset(to_idx=idx)
    env_train._orig_reset = env_train.reset
    env_train.reset = train_reset

    env_val = EnvKPI(train_csv, test_csv)
    env_val.datasetsize = total
    def val_reset(idx):
        return env_val.reset(to_idx=idx)
    env_val._orig_reset = env_val.reset
    env_val.reset = val_reset

    # 3) Session & estimators
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)

    qlearn = qlearn_constructor(scope="qlearn")
    target = target_constructor(scope="target")
    sess.run(tf.compat.v1.global_variables_initializer())

    best_val_f1 = 0.0
    checks_no_improve = 0

    episode_rewards = []
    coef_history    = []
    dynamic_coef = 10.0

    for ep in range(1, num_episodes+1):
        # train one episode on a randomly selected KPI
        env_train.rewardfnc = lambda ts,tc,a: RNNBinaryRewardFuc(ts,tc,a,vae_model,dynamic_coef)
        state = env_train.reset()
        env_train.states_list = env_train.get_states_list()
        # ... run your episode exactly as in q_learning, but only for 1 episode ...
        # at end compute episode_reward, update replay, do your num_epoches updates
        # (omitted here for brevity; reuse your existing loop body)

        episode_rewards.append(episode_reward)
        coef_history.append(dynamic_coef)

        # update dynamic_coef as before
        dynamic_coef = update_dynamic_coef_proportional(dynamic_coef, episode_reward,
                                                        target=0.0, alpha=0.001)

        # every validate_every episodes, run validation pass
        if ep % validate_every == 0:
            f1s = []
            for idx in val_indices:
                # test each KPI once
                env_val.rewardfnc = RNNBinaryRewardFucTest
                state = env_val.reset(to_idx=idx)
                env_val.states_list = env_val.get_states_list()
                # run 1 greedy episode
                preds, truths = [], []
                while True:
                    probs = make_epsilon_greedy_policy(qlearn, env_val.action_space_n, sess)(state, 0)
                    act = np.argmax(probs)
                    preds.append(act)
                    truths.append(env_val.timeseries['anomaly'].iat[env_val.timeseries_cursor])
                    nxt, _, done, _ = env_val.step(act)
                    if done: break
                    state = nxt[act]
                p, r, f, _ = precision_recall_fscore_support(truths, preds,
                                        average='binary', zero_division=0)
                f1s.append(f)
            val_f1 = np.mean(f1s)
            print(f"Validation F1 @ ep {ep}: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                checks_no_improve = 0
                # optionally save a checkpoint of your Q-network here
            else:
                checks_no_improve += 1
                if checks_no_improve >= patience:
                    print(f"No improvement for {patience} checks—early stopping at ep {ep}.")
                    break

    # after training, you can run full test on env_full_test if desired
    return episode_rewards, coef_history, best_val_f1