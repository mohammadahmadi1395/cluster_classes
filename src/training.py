from src import utility_functions
import keras
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
from math import pi


class L2Normalization(keras.layers.Layer):
    """This layer normalizes the inputs with l2 normalization."""

    def __init__(self, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs):
        inputs = tf.nn.l2_normalize(inputs, axis=1)

        return inputs

    def get_config(self):
        config = super().get_config()
        return config

class ArcLayer(keras.layers.Layer):
    """
    Custom layer for ArcFace.

    This layer is equivalent a dense layer except the weights are normalized.
    """

    def __init__(self, units, kernel_regularizer=None, **kwargs):
        super(ArcLayer, self).__init__(**kwargs)
        self.units = units
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=[input_shape[-1], self.units],
                                      dtype=tf.float32,
                                      initializer=keras.initializers.HeNormal(),
                                      regularizer=self.kernel_regularizer,
                                      trainable=True,
                                      name='kernel')
        self.built = True

    @tf.function
    def call(self, inputs):
        weights = tf.nn.l2_normalize(self.kernel, axis=0)
        return tf.matmul(inputs, weights)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units,
                       "kernel_regularizer": self.kernel_regularizer})
        return config

class ArcLoss(tf.keras.losses.Loss):
    """Additive angular margin loss.
    Original implementation: https://github.com/luckycallor/InsightFace-tensorflow
    """

    def __init__(self, margin=0.5, scale=64, name="arcloss", n_classes=500):
        """Build an additive angular margin loss object for Keras model."""
        super().__init__(name=name)
        self.margin = margin
        self.scale = scale
        self.threshold = tf.math.cos(pi - margin)
        self.cos_m = tf.math.cos(margin)
        self.sin_m = tf.math.sin(margin)
        self.n_classes = n_classes

        # Safe margin: https://github.com/deepinsight/insightface/issues/108
        self.safe_margin = self.sin_m * margin

    @tf.function
    def call(self, y_true, y_pred):
        # Calculate the cosine value of theta + margin.
        cos_t = y_pred
        sin_t = tf.math.sqrt(1 - tf.math.square(cos_t))

        cos_t_margin = tf.where(cos_t > self.threshold,
                                cos_t * self.cos_m - sin_t * self.sin_m,
                                cos_t - self.safe_margin)

        # The labels here had already been onehot encoded.
        # y_true = tf.one_hot(tf.cast(y_true, tf.int64), depth=self.n_classes)
        mask = tf.cast(y_true, tf.float32)
        cos_t_onehot = cos_t * mask
        cos_t_margin_onehot = cos_t_margin * mask

        # Calculate the final scaled logits.
        logits = (cos_t + cos_t_margin_onehot - cos_t_onehot) * self.scale
        # logits = (cos_t + cos_t_margin - cos_t) * self.scale

        losses = tf.nn.softmax_cross_entropy_with_logits(y_true, logits)
        # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, logits)

        return losses

    def get_config(self):
        config = super(ArcLoss, self).get_config()
        config.update({"margin": self.margin, "scale": self.scale})
        return config

"""This module provides the implementation of training supervisor."""


class TrainingSupervisor(object):
    """A training supervisor will organize and monitor the training process."""

    def __init__(self, model, optimizer, loss, dataset, training_dir, save_freq, monitor, mode, name, num_ids) -> None: #, cluster_idx, dataset_name, scenario_number, test_dataset) -> None:
        """Training supervisor organizes and monitors the training process.

        Args:
            model: the Keras model to be trained.
            optimizer: a Keras optimizer used for training.
            loss: a Keras loss function.
            dataset: the training dataset.
            training_dir: the directory to save the training files.
            save_freq: integer, the supervisor saves the model at end of this many batches.
            monitor: the metric name to monitor.
            mode: one of {'min', 'max'}
            name: current model or project name.
        """
        super().__init__()

        # Track the objects used for training.
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss
        self.dataset = dataset
        self.data_generator = iter(self.dataset)
        self.save_freq = save_freq
        self.metrics = {
            'categorical_accuracy': tf.keras.metrics.CategoricalAccuracy(
                name='train_accuracy', dtype=tf.float32),
            'loss': tf.keras.metrics.Mean(name="train_loss_mean",
                                          dtype=tf.float32),
            'val_loss': tf.keras.metrics.Mean(name='val_loss_mean', dtype=tf.float32),
            'val_acc': tf.keras.metrics.SparseCategoricalAccuracy(
                name='validation_accuracy', dtype=tf.float32)
            }

        self.monitor = self.metrics[monitor]
        self.mode = mode
        self.num_ids = num_ids
        self.training_dir = training_dir
        # Training schedule tracks the training progress. The training
        # supervisor uses this object to make training arrangement. The schedule
        # is saved in the checkpoint and maintained by the manager.
        self.schedule = {
            'step': tf.Variable(0, trainable=False, dtype=tf.int64),
            'epoch': tf.Variable(1, trainable=False, dtype=tf.int64),
            'monitor_value': tf.Variable(0, trainable=False, dtype=tf.float32)}

        # Both the model and the training status shall be tracked. A TensorFlow
        # checkpoint is the best option to fullfill this job.
        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            metrics=self.metrics,
            schedule=self.schedule,
            monitor=self.monitor,
            dataset=self.data_generator)

        # A model manager is responsible for saving the current training
        # schedule and the model weights.
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            os.path.join(training_dir, 'checkpoints', name),
            max_to_keep=2)

        # A model scout watches and saves the best model according to the
        # monitor value.
        self.scout = tf.train.CheckpointManager(
            self.checkpoint,
            os.path.join(training_dir, 'model_scout', name),
            max_to_keep=1)

        # A clerk writes the training logs to the TensorBoard.
        self.clerk = tf.summary.create_file_writer(
            os.path.join(training_dir, 'logs', name))

    def restore(self, weights_only=False, from_scout=False):
        """Restore training process from previous training checkpoint.

        Args:
            weights_only: only restore the model weights. Default is False.
            from_scout: restore from the checkpoint saved by model scout.
        """
        # Are there any checkpoint files?
        if from_scout:
            latest_checkpoint = self.scout.latest_checkpoint
        else:
            latest_checkpoint = self.manager.latest_checkpoint

        if latest_checkpoint:
            # utility_functions.pprint(("Checkpoint found: {}".format(latest_checkpoint)))
            print("Checkpoint found: {}".format(latest_checkpoint))
        else:
            print("WARNING: Checkpoint not found. Model will be initialized from scratch.")

        if weights_only:
            print("Only the model weights will be restored.")
            checkpoint = tf.train.Checkpoint(self.model)
            checkpoint.restore(latest_checkpoint)
        else:
            self.checkpoint.restore(latest_checkpoint)

        print("Checkpoint restored: {}".format(latest_checkpoint))

    @tf.function
    def _train_step(self, x_batch, y_batch):
        """Define the training step function.

        Args:
            x_batch: the inputs of the network.
            y_batch: the labels of the batched inputs.

        Returns:
            logtis and loss.
        """

        with tf.GradientTape() as tape:
            # Run the forward propagation.
            logits = self.model(x_batch, training=True)

            # Calculate the loss value from targets and regularization.
            loss = self.loss_fun(y_batch, logits) + sum(self.model.losses)

        # Calculate the gradients.
        grads = tape.gradient(loss, self.model.trainable_weights)

        # Back propagation.
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))

        return logits, loss

    @tf.function
    def _val_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            # Run the forward propagation.
            logits = self.model(x_batch, training=False)

            # Calculate the loss value from targets and regularization.
            loss = self.loss_fun(y_batch, logits) + sum(self.model.losses)

        return logits, loss


    @tf.function
    def _update_metrics(self, labels, logits, loss):
        """Update the metrics.

        Args:
            labels: the labels of the batched inputs.
            logits: the outputs of the model.
            loss: the loss value of current training step.
        """
        self.metrics['categorical_accuracy'].update_state(labels, logits)
        self.metrics['loss'].update_state(loss)

    def _update_val_metrics(self, val_labels, val_logits, val_loss):
        self.metrics['val_loss'].update_state(val_loss)
        self.metrics['val_acc'].update_state(val_labels, val_logits)
    

    def _reset_metrics(self):
        """Reset all the metrics."""
        for _, metric in self.metrics.items():
            metric.reset_states

    def _log_to_tensorboard(self):
        """Log the training process to TensorBoard."""
        # Get the parameters to log.
        current_step = int(self.schedule['step'])
        train_loss = self.metrics['loss'].result()
        train_acc = self.metrics['categorical_accuracy'].result()
        val_loss = self.metrics['val_loss'].result()
        val_acc = self.metrics['val_acc'].result()
        lr = self.optimizer._decayed_lr('float32')

        with self.clerk.as_default():
            tf.summary.scalar("loss", train_loss,   step=current_step)
            tf.summary.scalar("accuracy", train_acc, step=current_step)
            tf.summary.scalar("learning rate", lr, step=current_step)
            tf.summary.scalar('val_loss', val_loss, step=current_step)
            tf.summary.scalar('val_acc', val_acc, step=current_step)


        # Log to STDOUT.
        print("Training accuracy: {:.4f}, mean loss: {:.2f}".format( # mean_val_loss: {:.4f}, val_acc: {:.4f}".format(
            float(train_acc), float(train_loss)))#, float(val_loss), float(val_acc))))

    def _checkpoint(self,epoch_idx):
        """Checkpoint the current training process.

        Args:
        monitor: the metric value to monitor.
        mode: one of {'min', 'max'}
        """
        # A helper function to check values by mode.
        def _check_value(v1, v2, mode):
            if (v1 < v2) & (mode == 'min'):
                return True
            elif (v1 > v2) & (mode == 'max'):
                return True
            else:
                return False

        # Get previous and current monitor values.
        previous = self.schedule['monitor_value'].numpy()
        current = self.monitor.result()

        # For the first checkpoint, initialize the monitor value to make
        # subsequent comparisons valid.
        if previous == 0.0:
            self.schedule['monitor_value'].assign(current)

        # Is current model the best one we had ever seen?
        if _check_value(current, previous, self.mode):
            # print("Monitor value improved from {:.4f} to {:.4f}."
            #       .format(previous, current))

            # Update the schedule.
            self.schedule['monitor_value'].assign(current)

            # And save the model.
            best_model_path = self.scout.save()
            # print("Best model found and saved: {}".format(best_model_path))

        # Save a regular checkpoint.
        self._reset_metrics()
        ckpt_path = self.manager.save()
        os.makedirs(os.path.join(self.training_dir, 'exported', 'hrnetv2', str(epoch_idx)),exist_ok=True)
        self.model.save(os.path.join(self.training_dir, 'exported', 'hrnetv2', str(epoch_idx)))

    def train(self, epochs, steps_per_epoch):
        """Train the model for epochs.

        Args:
            epochs: an integer number of epochs to train the model.
            steps_per_epoch: an integer numbers of steps for one epoch.
        """
        # In case the training is resumed, where are now?
        initial_epoch = self.schedule['epoch'].numpy()
        global_step = self.schedule['step'].numpy()
        initial_step = global_step % steps_per_epoch

        # Start training loop.
        for epoch in range(initial_epoch, epochs + 1):
            # Log current epoch.
            print("Epoch {}/{}".format(epoch, epochs))

            # Visualize the training progress.
            progress_bar = tqdm(total=steps_per_epoch, initial=initial_step,
                                ascii="->", colour='#1cd41c')

            # Iterate over the batches of the dataset
            for x_batch, y_batch in self.data_generator:
                y_batch = tf.one_hot(y_batch, self.num_ids)
                # Train for one step.
                logits, loss = self._train_step(x_batch, y_batch)

                # Update the metrics.
                # self._update_metrics(tf.expand_dims(y_batch, 1), logits, loss)
                self._update_metrics(y_batch, logits, loss)

                # Update the training schedule.
                self.schedule['step'].assign_add(1)

                # Update the progress bar.
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": "{:.2f}".format(loss.numpy()),
                    "accuracy": "{:.3f}".format(
                        self.metrics['categorical_accuracy'].result().numpy())})

                # Log and checkpoint the model.
                if int(self.schedule['step']) % self.save_freq == 0:
                    pass
           
            # Update the checkpoint epoch counter.
            self.schedule['epoch'].assign_add(1)

            # Reset the training dataset.
            self.data_generator = iter(self.dataset)

            # Clean up the progress bar.
            progress_bar.close()

        print("Training accomplished at epoch {}".format(epochs))


    def export(self, model, export_dir):
        """Export the model in saved_model format.

        Args:
            export_dir: the direcotry where the model will be saved.
        """
        model.save(export_dir)

    def override(self, step=None, epoch=None, monitor_value=None):
        """Override the current training schedule with a new one.

        The parameter won't be overridden if new value is None.

        Args:
            step: new training step to start from.
            epoch: new epoch to start from.
            monitor_value: new monitor value to start with.
        """
        if step:
            self.schedule['step'].assign(step)

        if epoch:
            self.schedule['epoch'].assign(epoch)

        if monitor_value:
            self.schedule['monitor_value'].assign(monitor_value)

def softmax_train(dataset_name, model_scenario_path, train_dataset, cluster, trainx, trainl, sub_index, freq=1000, epochs=50, train_overwrite=False):
    """
    Train a softmax-based model for classification.

    Args:
        dataset_name (str): Name of the dataset.
        model_scenario_path (str): Path to the model scenario.
        train_dataset (keras.utils.data.Dataset): Training dataset.
        cluster (list): List of clusters.
        trainx (numpy.ndarray): Training input data.
        trainl (numpy.ndarray): Training labels.
        sub_index (int): Sub-index for model scenario.
        freq (int, optional): Frequency of training checkpoints. Default is 1000.
        epochs (int, optional): Number of training epochs. Default is 50.
        train_overwrite (bool, optional): Whether to overwrite existing training data. Default is False.

    Returns:
        keras.Model: The trained softmax-based model.
    """

    frequency = freq
    name = "hrnetv2"
    export_dir = os.path.join(model_scenario_path, str(sub_index), 'exported', name)

    input_shape = (512,)
    num_ids = len(cluster)
    regularizer = keras.regularizers.L2(5e-4)
    name = "hrnetv2"

    # Define the model architecture
    model = keras.Sequential([keras.Input(input_shape), \
        L2Normalization(), \
        ArcLayer(num_ids, regularizer)], \
        name="training_model")

    # Define the loss function
    loss_fun = ArcLoss(n_classes=num_ids)
    
    # Define the optimizer
    optimizer = keras.optimizers.Adam(0.001, amsgrad=True, epsilon=0.001)

    path = None
    path = os.path.join(model_scenario_path, str(sub_index))

    # Create a TrainingSupervisor
    supervisor = TrainingSupervisor(model,
                                    optimizer,
                                    loss_fun,
                                    train_dataset,
                                    path,
                                    frequency,
                                    "categorical_accuracy",
                                    'max',
                                    name,
                                    num_ids)
    model.summary()

    # Restore the model if available
    supervisor.restore(True)

    # Train the model
    supervisor.train(epochs, freq)

    # Export the trained model
    supervisor.export(model, export_dir)

    return model
