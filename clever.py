import tensorflow as tf


def clever_t(
        model,
        example,
        predicted_label: int,
        target_label: int,
        batch_size: int,
        samples_per_batch: int,
        perturbation_norm: int,
        maximum_perturbation: float,
    ) -> float:
    """
    Implements the targeted CLEVER score.
    
    TODO: I'm not sure how to fit a reverse-weibull distribution to the data, to get the location parameter A_W.
    Right now I'm just applying gradient descent with an optimizer and trying to minimize the MSE between the data and what comes out of the reverse weibull distribution.
    """
    x_0 = example
    q = perturbation_norm // (perturbation_norm - 1)

    @tf.function
    def g(x):
        f_x = model(x, training=True)
        return f_x[:,predicted_label] - f_x[:,target_label]
    
    @tf.function
    def random_point_in_ball(center, radius):
        noise = tf.random.uniform(shape=center.shape, minval=-radius, maxval=+radius)
        out = center + noise
        return out
    
    def populate_S():
        S = tf.TensorArray(size=batch_size, dtype=tf.float32)
        x_batch = tf.tile(x_0, [samples_per_batch,1,1,1])

        for i in range(batch_size):
            x_perturbed = random_point_in_ball(x_batch, maximum_perturbation)
            with tf.GradientTape() as tape:
                tape.watch(x_perturbed)
                g_x = g(x_perturbed)

                #Compute and flatten the gradient
                grad_g = tape.gradient(g_x, x_perturbed)
                grad_g = tf.reshape(grad_g, [samples_per_batch, -1])

                norm = tf.norm(grad_g, axis=-1, ord=q)
                max_g = tf.reduce_max(norm)
                # Write the result
                S.write(i, max_g)

        S = S.stack()
        return S

    def fit_reverse_weibull_distribution(max_g_gradients, iterations = 100):
        """
        #TODO: I Don't know how to "properly" fit a reverse weibull distribution on the data.
        """
        scale = tf.Variable(5.0)
        location = tf.Variable(0.0)
        shape = tf.Variable(1.0)

        @tf.function
        def reverse_weibull_gap():
            x = max_g_gradients
            y = tf.random.uniform(x.shape)
            weibull_x = tf.exp(-((location - y)/ scale)**shape)
            loss = tf.reduce_mean((x - weibull_x) ** 2)
            #print(weibull_x)
            #print(x)
            #print(tf.reduce_mean(weibull_x - x))
            #print("loss:", loss)
            return loss

        loss = reverse_weibull_gap 

        optimizer = tf.keras.optimizers.SGD()
        for i in range(iterations):
            grad = optimizer.minimize(loss, var_list=[scale, location, shape])
#             if tf.executing_eagerly():
#                 print(i, "\tscale:", scale.numpy(), "location:", location.numpy(), "shape:", shape.numpy())
        return location
    
    S = populate_S()
    location_estimate = fit_reverse_weibull_distribution(S, iterations=100)
    clever_score = min(g(x_0) / location_estimate, maximum_perturbation)
    return tf.reshape(clever_score, ())
    
def clever_u(
        model,
        example,
        predicted_label: int,
        num_labels: int,
        batch_size: int,
        samples_per_batch: int,
        perturbation_norm: int,
        maximum_perturbation: float,
    ) -> float:
    results = tf.TensorArray(size=num_labels-1, dtype=tf.float32)
    for i, target_label in enumerate(filter(lambda label: label != predicted_label, range(num_labels))):
        clever_for_label = clever_t(
            model,
            example,
            predicted_label,
            target_label,
            batch_size,
            samples_per_batch,
            perturbation_norm,
            maximum_perturbation
        )
        results.write(i, clever_for_label)
    results = results.stack()
    
    return tf.reduce_min(results)
    